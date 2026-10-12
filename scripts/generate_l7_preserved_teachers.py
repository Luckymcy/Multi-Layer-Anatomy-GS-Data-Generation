"""Composite a fat mesh with frozen L6 Gaussian RGB/alpha/depth into L7 targets.

The Gaussian depth is alpha-normalized expected inverse depth, so occlusion
at translucent intersections is approximate. Existing L6 is kept as rendered,
avoiding a different reconstructed heart mesh in the learning target.
"""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import vtk
from vtk.util.numpy_support import vtk_to_numpy

import generate_heart_exterior_teachers_v4 as exterior
import generate_heart_teachers_v2 as legacy


def capture(window, kind):
    grab = vtk.vtkWindowToImageFilter()
    grab.SetInput(window)
    if kind == 'rgba':
        grab.SetInputBufferTypeToRGBA()
    else:
        grab.SetInputBufferTypeToZBuffer()
    grab.ReadFrontBufferOff()
    grab.Update()
    width, height = window.GetSize()
    data = vtk_to_numpy(grab.GetOutput().GetPointData().GetScalars())
    return data.reshape((height, width, 4) if kind == 'rgba' else (height, width))[::-1].copy()


def downsample(array, size):
    if array.ndim == 2:
        return np.asarray(Image.fromarray(array.astype(np.float32)).resize((size, size), Image.Resampling.LANCZOS))
    return np.stack([downsample(array[..., i], size) for i in range(array.shape[2])], axis=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True)
    parser.add_argument('--mesh', required=True)
    parser.add_argument('--source-teachers', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    base, output = Path(args.base), Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    images = output / 'images'
    images.mkdir(parents=True)
    cameras = json.loads((base / 'cameras.json').read_text())
    manifest = json.loads((Path(args.source_teachers).parent / 'manifest.json').read_text())
    center = np.asarray(manifest['camera']['center'])
    radius = manifest['camera']['radius']
    reader = vtk.vtkPLYReader()
    reader.SetFileName(args.mesh)
    reader.Update()
    surface = reader.GetOutput()
    color = (0.84, 0.70, 0.47)
    actor = exterior.material_actor(surface, color, 0.58, 0.30)
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0, 0, 0)
    renderer.SetBackgroundAlpha(0)
    exterior.add_lights(renderer, center, radius)
    renderer.AddActor(actor)
    window = vtk.vtkRenderWindow()
    window.SetOffScreenRendering(1)
    window.SetAlphaBitPlanes(1)
    window.SetMultiSamples(0)
    window.SetSize(cameras[0]['width'], cameras[0]['height'])
    window.AddRenderer(renderer)
    camera = renderer.GetActiveCamera()
    poses, review_paths, visible_counts = [], [], []
    for index, record in enumerate(cameras):
        position = np.asarray(record['position'])
        target = position + np.asarray(record['forward']) * radius
        camera.SetPosition(*position)
        camera.SetFocalPoint(*target)
        camera.SetViewUp(*record['up'])
        camera.SetViewAngle(record['fov_y_degrees'])
        camera.SetParallelProjection(False)
        renderer.ResetCameraClippingRange()
        window.Render()
        rgba = capture(window, 'rgba').astype(np.float32) / 255
        z_buffer = capture(window, 'z')
        near, far = camera.GetClippingRange()
        fat_depth = near * far / (far - z_buffer * (far - near))
        data = np.load(base / (record['name'] + '.npz'))
        base_rgb = data['rgb'].astype(np.float32)
        base_alpha = data['alpha'].astype(np.float32)
        base_depth = data['depth']
        a = rgba[..., 3]
        # No multisampling: opaque mesh pixels have alpha 1. All blending
        # and supersampling happen in premultiplied color below.
        fat_rgb = rgba[..., :3] * a[..., None]
        foreground = np.clip((base_depth - fat_depth) / 0.4 + 0.5, 0, 1)
        front_rgb = fat_rgb + base_rgb * (1 - a[..., None])
        back_rgb = base_rgb + fat_rgb * (1 - base_alpha[..., None])
        combined = front_rgb * foreground[..., None] + back_rgb * (1 - foreground[..., None])
        alpha = a + base_alpha * (1 - a)
        alpha_out = np.clip(downsample(alpha, 512), 0, 1)
        rgb_out = downsample(combined, 512)
        rgb_out = np.divide(rgb_out, alpha_out[..., None], out=np.zeros_like(rgb_out),
                            where=alpha_out[..., None] > 1e-6)
        target_rgba = np.concatenate((np.clip(rgb_out, 0, 1), alpha_out[..., None]), axis=2)
        path = images / (record['name'] + '.png')
        Image.fromarray(np.round(target_rgba * 255).astype(np.uint8), 'RGBA').save(path)
        poses.append((position, target))
        visible_counts.append(int(((a > 0.5) & ((foreground > 0.5) | (base_alpha < 0.5))).sum()))
        if index in (20, 70, 100):
            base_a = np.clip(downsample(base_alpha, 512), 0, 1)
            base_straight = downsample(base_rgb, 512) / np.maximum(base_a[..., None], 1e-6)
            base_rgba = np.concatenate((np.clip(base_straight, 0, 1), base_a[..., None]), axis=2)
            before = output / f'base_{index:03}.png'
            Image.fromarray(np.round(base_rgba * 255).astype(np.uint8), 'RGBA').save(before)
            comparison = output / f'preview_{index:03}.png'
            exterior.write_comparison([str(before), str(path)], ['Frozen L6', 'L7 fat teacher'], str(comparison))
            review_paths.append(str(comparison))
        if (index + 1) % 20 == 0:
            print(f'Rendered {index + 1}/{len(cameras)} L7 targets', flush=True)
    points = legacy.sample_points(surface, 60000, 49)
    legacy.write_colmap(str(output), poses, 512, 512, cameras[0]['fov_y_degrees'], points, color)
    for name in ('train.txt', 'test.txt'):
        shutil.copyfile(Path(args.source_teachers) / name, output / name)
    info = {
        'layer': 7, 'name': 'heart_adjacent_fat_preserved', 'mesh': args.mesh,
        'base': args.base, 'views': len(cameras), 'initial_points': len(points),
        'fat_color_rgb_0_1': color, 'visible_fat_pixels_highres_percentiles': np.percentile(visible_counts, [5, 50, 95]).tolist(),
        'composition': 'Frozen L6 premultiplied RGB/alpha + fat mesh; expected inverse-depth occlusion with 0.4 mm transition.',
        'previews': review_paths,
    }
    (output / 'manifest.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    window.Finalize()


if __name__ == '__main__':
    main()
