import argparse
import json
import os
import shutil
import sys

import numpy as np
import SimpleITK as sitk
from PIL import Image
import vtk
from vtk.util.numpy_support import vtk_to_numpy


sys.path.append(os.path.dirname(__file__))
import generate_heart_teachers_v2 as legacy
import generate_heart_exterior_teachers_v4 as exterior


NEW_LAYER_SPECS = [
    (4, "myocardium", (0.56, 0.105, 0.085)),
    (5, "great_vessels", (0.68, 0.145, 0.115)),
    (6, "coronary_arteries", (0.30, 0.025, 0.030)),
]


def surface_from_mask(mask, sigma_mm, iterations, pass_band):
    sdf = sitk.SignedMaurerDistanceMap(
        sitk.Cast(mask > 0, sitk.sitkUInt8),
        insideIsPositive=True,
        squaredDistance=False,
        useImageSpacing=True,
    )
    sdf = sdf + min(mask.GetSpacing()) * 0.5
    if sigma_mm > 0:
        sdf = sitk.SmoothingRecursiveGaussian(sdf, sigma_mm)
    return surface_from_sdf(sdf, iterations, pass_band)


def surface_from_sdf(sdf, iterations, pass_band):
    contour = vtk.vtkFlyingEdges3D()
    contour.SetInputData(
        exterior.vtk_image_from_sitk(sitk.Cast(sdf, sitk.sitkFloat32))
    )
    contour.SetValue(0, 0.0)
    contour.ComputeNormalsOff()
    contour.Update()

    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputConnection(contour.GetOutputPort())
    smoother.SetNumberOfIterations(iterations)
    smoother.SetPassBand(pass_band)
    smoother.BoundarySmoothingOff()
    smoother.FeatureEdgeSmoothingOff()
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.Update()

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(smoother.GetOutputPort())
    normals.ComputePointNormalsOn()
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.SplittingOff()
    normals.Update()
    output = vtk.vtkPolyData()
    output.DeepCopy(normals.GetOutput())
    return output


def build_actor(surface, color, layer_index):
    roughness = 0.50 if layer_index == 4 else 0.58
    specular = 0.42 if layer_index == 4 else 0.30
    return exterior.material_actor(
        surface,
        color=color,
        roughness=roughness,
        specular=specular,
    )


def capture_downsampled(renderer, window, camera, pose, view_angle,
                        output_path, output_size):
    legacy.configure_camera(camera, pose[0], pose[1], view_angle)
    renderer.ResetCameraClippingRange()
    window.Render()
    capture = vtk.vtkWindowToImageFilter()
    capture.SetInput(window)
    capture.SetInputBufferTypeToRGBA()
    capture.ReadFrontBufferOff()
    capture.Update()
    vtk_output = capture.GetOutput()
    width, height = window.GetSize()
    pixels = vtk_to_numpy(vtk_output.GetPointData().GetScalars())
    pixels = pixels.reshape((height, width, 4))[::-1]
    image = Image.fromarray(pixels, mode="RGBA")
    if width != output_size or height != output_size:
        image = image.resize(
            (output_size, output_size), Image.Resampling.LANCZOS
        )
    image.save(output_path)


def write_split(output, pose_count, base_view_count):
    test_indices = list(range(0, base_view_count, 8))
    test_set = set(test_indices)
    train_indices = [index for index in range(pose_count) if index not in test_set]
    with open(os.path.join(output, "train.txt"), "w", encoding="ascii") as file:
        file.write("\n".join(f"view_{index:06}.png" for index in train_indices))
    with open(os.path.join(output, "test.txt"), "w", encoding="ascii") as file:
        file.write("\n".join(f"view_{index:06}.png" for index in test_indices))


def render_layer(output, actors, poses, active_surface, active_color,
                 center, radius, output_size, render_size, view_angle,
                 point_count, layer_index, base_view_count):
    images_dir = os.path.join(output, "images")
    os.makedirs(images_dir, exist_ok=True)
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0.0, 0.0, 0.0)
    renderer.SetBackgroundAlpha(0.0)
    exterior.add_lights(renderer, center, radius)
    window = vtk.vtkRenderWindow()
    window.SetOffScreenRendering(1)
    window.SetAlphaBitPlanes(1)
    window.SetMultiSamples(8)
    window.SetSize(render_size, render_size)
    window.AddRenderer(renderer)
    for actor in actors:
        renderer.AddActor(actor)
    camera = renderer.GetActiveCamera()

    for index, pose in enumerate(poses):
        capture_downsampled(
            renderer,
            window,
            camera,
            pose,
            view_angle,
            os.path.join(images_dir, f"view_{index:06}.png"),
            output_size,
        )
        if (index + 1) % 20 == 0 or index + 1 == len(poses):
            print(f"L{layer_index}: rendered {index + 1}/{len(poses)} views", flush=True)

    points = legacy.sample_points(
        active_surface, point_count, 42 + layer_index
    )
    legacy.write_colmap(
        output,
        poses,
        output_size,
        output_size,
        view_angle,
        points,
        active_color,
    )
    write_split(output, len(poses), base_view_count)
    renderer.RemoveAllViewProps()
    window.Finalize()
    return len(points)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build the V5 heart teachers: reuse L0-L3 and render isotropic "
            "SDF-reconstructed L4-L6."
        )
    )
    parser.add_argument("--existing-teachers", required=True)
    parser.add_argument("--l4-outer-mask", required=True)
    parser.add_argument("--l5-mask", required=True)
    parser.add_argument("--l6-sdf", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--render-size", type=int, default=1024)
    parser.add_argument("--output-size", type=int, default=512)
    parser.add_argument("--point-count", type=int, default=60000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if os.path.exists(args.output_dir):
        if not args.overwrite:
            raise FileExistsError(
                f"Output already exists: {args.output_dir}. Use --overwrite."
            )
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir)

    with open(
        os.path.join(args.existing_teachers, "manifest.json"),
        "r",
        encoding="utf-8",
    ) as file:
        old_manifest = json.load(file)

    for layer_index in range(4):
        shutil.copytree(
            os.path.join(args.existing_teachers, f"L{layer_index}"),
            os.path.join(args.output_dir, f"L{layer_index}"),
        )

    l4_mask = sitk.ReadImage(args.l4_outer_mask)
    l5_mask = sitk.ReadImage(args.l5_mask)
    l6_sdf = sitk.ReadImage(args.l6_sdf)
    l4_surface = surface_from_mask(l4_mask, 1.00, 12, 0.10)
    l5_surface = surface_from_mask(l5_mask, 0.80, 12, 0.10)
    l6_surface = surface_from_sdf(
        sitk.SmoothingRecursiveGaussian(l6_sdf, 0.16), 4, 0.22
    )
    surfaces = [l4_surface, l5_surface, l6_surface]
    actors = [
        build_actor(surface, spec[2], spec[0])
        for surface, spec in zip(surfaces, NEW_LAYER_SPECS)
    ]

    camera_config = old_manifest["camera"]
    center = np.asarray(camera_config["center"], dtype=float)
    radius = float(camera_config["radius"])
    view_angle = float(camera_config["view_angle_degrees"])
    base_count = int(camera_config["base_views"])
    extra_count = int(camera_config["extra_coronary_views"])
    base_poses = legacy.make_poses(center, radius, base_count, 42)
    coronary_min, coronary_max = legacy.surface_bounds(l6_surface)
    coronary_center = (coronary_min + coronary_max) * 0.5
    coronary_radius = max(
        float(np.linalg.norm(coronary_max - coronary_center)), 1.0
    )
    extra_poses = legacy.make_poses(
        coronary_center,
        max(radius, coronary_radius),
        extra_count,
        1042,
    )

    new_entries = []
    for offset, (layer_index, name, color) in enumerate(NEW_LAYER_SPECS):
        poses = list(base_poses)
        if layer_index == 6:
            poses.extend(extra_poses)
        # L0-L3 still exist as independently controllable layers, but once L4
        # is enabled they are physically hidden by its continuous exterior.
        visible_actors = actors[: offset + 1]
        output = os.path.join(args.output_dir, f"L{layer_index}")
        sampled_count = render_layer(
            output,
            visible_actors,
            poses,
            surfaces[offset],
            color,
            center,
            radius,
            args.output_size,
            args.render_size,
            view_angle,
            args.point_count,
            layer_index,
            base_count,
        )
        new_entries.append(
            {
                "index": layer_index,
                "name": name,
                "source": [
                    args.l4_outer_mask,
                    args.l5_mask,
                    args.l6_sdf,
                ][offset],
                "surface_method": "0.31 mm isotropic signed-distance field",
                "color_rgb_0_1": list(color),
                "color_rgb_8bit": np.round(
                    np.asarray(color) * 255
                ).astype(int).tolist(),
                "views": len(poses),
                "initial_points": sampled_count,
                "teacher_visible_layers": list(range(4, layer_index + 1)),
            }
        )

    manifest = {
        "generator": "generate_heart_teachers_v5_sdf.py",
        "based_on": os.path.join(args.existing_teachers, "manifest.json"),
        "camera": camera_config,
        "split": old_manifest["split"],
        "rendering": {
            "render_size": args.render_size,
            "output_size": args.output_size,
            "downsample": "Lanczos",
            "material": "PBR",
            "L4_sdf_sigma_mm": 1.0,
            "L5_sdf_sigma_mm": 0.8,
            "L6_sdf_sigma_mm": 0.16,
        },
        "layers": old_manifest["layers"][:4] + new_entries,
    }
    with open(
        os.path.join(args.output_dir, "manifest.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(manifest, file, indent=2)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
