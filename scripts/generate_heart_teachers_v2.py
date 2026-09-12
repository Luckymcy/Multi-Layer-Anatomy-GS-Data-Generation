import argparse
import json
import os
import shutil
import sys

import numpy as np
import SimpleITK as sitk
from PIL import Image
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

sys.path.append(os.path.dirname(__file__))
import read_write_model as colmap


LAYER_SPECS = [
    ("L0_left_ventricle.nii.gz", "left_ventricle", (0.90, 0.15, 0.15)),
    ("L1_right_ventricle.nii.gz", "right_ventricle", (0.15, 0.40, 0.95)),
    ("L2_left_atrium.nii.gz", "left_atrium", (0.95, 0.40, 0.65)),
    ("L3_right_atrium.nii.gz", "right_atrium", (0.15, 0.80, 0.85)),
    ("L4_myocardium.nii.gz", "myocardium", (0.65, 0.12, 0.22)),
    ("L5_great_vessels.nii.gz", "great_vessels", (0.95, 0.55, 0.10)),
    ("L6_coronary_arteries.nii.gz", "coronary_arteries", (1.00, 0.90, 0.15)),
]


def read_mask(path):
    image = sitk.ReadImage(path)
    return image, sitk.GetArrayFromImage(image).astype(bool)


def mask_to_vtk(image, array):
    vtk_image = vtk.vtkImageData()
    vtk_image.SetDimensions(image.GetSize())
    vtk_image.SetSpacing(image.GetSpacing())
    vtk_image.SetOrigin(image.GetOrigin())
    vtk_image.SetDirectionMatrix(image.GetDirection())
    vtk_image.GetPointData().SetScalars(
        numpy_to_vtk(
            array.astype(np.uint8).ravel(order="C"),
            deep=True,
            array_type=vtk.VTK_UNSIGNED_CHAR,
        )
    )
    return vtk_image


def extract_surface(image, array, iterations=8):
    marching = vtk.vtkMarchingCubes()
    marching.SetInputData(mask_to_vtk(image, array))
    marching.SetValue(0, 0.5)
    marching.ComputeNormalsOn()
    marching.Update()

    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputConnection(marching.GetOutputPort())
    smoother.SetNumberOfIterations(iterations)
    smoother.SetPassBand(0.12)
    smoother.BoundarySmoothingOff()
    smoother.FeatureEdgeSmoothingOff()
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.Update()
    return smoother.GetOutput()


def surface_bounds(surface):
    bounds = surface.GetBounds()
    minimum = np.array([bounds[0], bounds[2], bounds[4]], dtype=float)
    maximum = np.array([bounds[1], bounds[3], bounds[5]], dtype=float)
    return minimum, maximum


def make_poses(center, radius, count, seed):
    rng = np.random.default_rng(seed)
    indices = np.arange(count, dtype=float) + 0.5
    z = 1.0 - 2.0 * indices / count
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    theta = golden_angle * indices
    directions = np.column_stack(
        (np.sqrt(1.0 - z * z) * np.cos(theta),
         np.sqrt(1.0 - z * z) * np.sin(theta), z)
    )
    directions += rng.normal(0.0, 0.002, directions.shape)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return [(center + direction * (radius * 2.4), center) for direction in directions]


def camera_view_up(position, target):
    forward = target - position
    forward /= np.linalg.norm(forward)
    up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(forward, up)) > 0.95:
        up = np.array([0.0, 1.0, 0.0])
    return up


def camera_rotation(position, target):
    forward = target - position
    forward /= np.linalg.norm(forward)
    up = camera_view_up(position, target)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    return np.vstack((right, down, forward))


def configure_camera(camera, position, target, view_angle):
    up = camera_view_up(position, target)
    camera.SetPosition(*position)
    camera.SetFocalPoint(*target)
    camera.SetViewUp(*up)
    camera.SetParallelProjection(False)
    camera.SetViewAngle(view_angle)


def build_actor(surface, color):
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(surface)
    # Marching Cubes carries mask scalars. Disable their lookup-table colors so
    # the explicit anatomical palette below is actually used.
    mapper.ScalarVisibilityOff()
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*color)
    actor.GetProperty().SetOpacity(1.0)
    actor.GetProperty().SetInterpolationToPhong()
    return actor


def render_png(renderer, window, camera, position, target, view_angle, path):
    configure_camera(camera, position, target, view_angle)
    renderer.ResetCameraClippingRange()
    window.Render()
    capture = vtk.vtkWindowToImageFilter()
    capture.SetInput(window)
    capture.SetInputBufferTypeToRGBA()
    capture.ReadFrontBufferOff()
    capture.Update()
    output = capture.GetOutput()
    width, height = window.GetSize()
    pixels = vtk_to_numpy(output.GetPointData().GetScalars())
    pixels = pixels.reshape((height, width, 4))[::-1]
    Image.fromarray(pixels, mode="RGBA").save(path)


def sample_points(surface, count, seed):
    points = vtk_to_numpy(surface.GetPoints().GetData()).astype(np.float64)
    if len(points) <= count:
        return points
    rng = np.random.default_rng(seed)
    return points[rng.choice(len(points), count, replace=False)]


def write_colmap(output, poses, width, height, view_angle, points, color):
    sparse = os.path.join(output, "sparse", "0")
    os.makedirs(sparse, exist_ok=True)
    focal_length = 0.5 * height / np.tan(np.deg2rad(view_angle * 0.5))
    cameras = {
        1: colmap.Camera(
            1,
            "SIMPLE_PINHOLE",
            width,
            height,
            np.array([focal_length, width / 2.0, height / 2.0]),
        )
    }
    images = {}
    for index, (position, target) in enumerate(poses, start=1):
        rotation = camera_rotation(position, target)
        images[index] = colmap.Image(
            index,
            colmap.rotmat2qvec(rotation),
            -rotation @ position,
            1,
            f"view_{index - 1:06}.png",
            np.empty((0, 2)),
            np.empty(0, dtype=np.int64),
        )
    rgb = np.round(np.asarray(color) * 255.0).astype(np.uint8)
    point_records = {
        index: colmap.Point3D(
            index,
            point,
            rgb,
            0.0,
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
        )
        for index, point in enumerate(points, start=1)
    }
    colmap.write_model(cameras, images, point_records, sparse, ext=".txt")


def render_layer(output, individual_surfaces, base_poses, extra_poses, width,
                 height, view_angle, layer_index, point_count):
    images_dir = os.path.join(output, "images")
    os.makedirs(images_dir, exist_ok=True)
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0.0, 0.0, 0.0)
    renderer.SetBackgroundAlpha(0.0)
    window = vtk.vtkRenderWindow()
    window.SetOffScreenRendering(1)
    window.SetAlphaBitPlanes(1)
    window.SetSize(width, height)
    window.AddRenderer(renderer)
    camera = renderer.GetActiveCamera()

    # A teacher for Li shows every anatomical structure from L0 through Li,
    # each as a separate surface with a stable layer-specific color.
    for visible_layer in range(layer_index + 1):
        renderer.AddActor(
            build_actor(
                individual_surfaces[visible_layer],
                LAYER_SPECS[visible_layer][2],
            )
        )

    poses = list(base_poses)
    if layer_index == 6:
        poses.extend(extra_poses)
    for index, (position, target) in enumerate(poses):
        render_png(
            renderer,
            window,
            camera,
            position,
            target,
            view_angle,
            os.path.join(images_dir, f"view_{index:06}.png"),
        )

    # The active Gaussian layer is initialized only from the newly introduced
    # anatomy, never from the cumulative union of previous layers.
    points = sample_points(individual_surfaces[layer_index], point_count, 42 + layer_index)
    write_colmap(
        output,
        poses,
        width,
        height,
        view_angle,
        points,
        LAYER_SPECS[layer_index][2],
    )

    # Use evenly distributed held-out base views. L6's extra coronary views
    # remain training-only so every layer shares the same 15-view test set.
    test_indices = list(range(0, len(base_poses), 8))
    test_index_set = set(test_indices)
    train_indices = [index for index in range(len(poses))
                     if index not in test_index_set]
    with open(os.path.join(output, "train.txt"), "w", encoding="ascii") as file:
        file.write("\n".join(f"view_{index:06}.png" for index in train_indices))
    with open(os.path.join(output, "test.txt"), "w", encoding="ascii") as file:
        file.write("\n".join(f"view_{index:06}.png" for index in test_indices))

    renderer.RemoveAllViewProps()
    window.Finalize()
    return len(poses), len(points)


def main():
    parser = argparse.ArgumentParser(
        description="Generate color-consistent, perspective heart teacher datasets."
    )
    parser.add_argument("--layers-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--views", type=int, default=120)
    parser.add_argument("--extra-coronary-views", type=int, default=60)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--view-angle", type=float, default=60.0)
    parser.add_argument("--point-count", type=int, default=60000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.views != 120:
        raise ValueError("The shared train/test split requires exactly 120 base views")
    if os.path.exists(args.output_dir):
        if not args.overwrite:
            raise FileExistsError(
                f"Output already exists: {args.output_dir}. Use a new path or --overwrite."
            )
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir)

    reference = None
    arrays = []
    for filename, _, _ in LAYER_SPECS:
        image, array = read_mask(os.path.join(args.layers_dir, filename))
        if reference is None:
            reference = image
        elif (image.GetSize() != reference.GetSize()
              or image.GetSpacing() != reference.GetSpacing()
              or image.GetOrigin() != reference.GetOrigin()
              or image.GetDirection() != reference.GetDirection()):
            raise ValueError(f"Geometry mismatch in {filename}")
        arrays.append(array)

    individual_surfaces = [
        extract_surface(reference, array.astype(np.uint8)) for array in arrays
    ]
    for index, surface in enumerate(individual_surfaces):
        if surface.GetNumberOfPoints() == 0:
            raise RuntimeError(f"No surface extracted for L{index}")

    full_union = np.logical_or.reduce(arrays)
    full_surface = extract_surface(reference, full_union.astype(np.uint8))
    bounds_min, bounds_max = surface_bounds(full_surface)
    center = (bounds_min + bounds_max) * 0.5
    radius = max(float(np.linalg.norm(bounds_max - center)), 1.0)
    base_poses = make_poses(center, radius, args.views, args.seed)

    coronary_min, coronary_max = surface_bounds(individual_surfaces[6])
    coronary_center = (coronary_min + coronary_max) * 0.5
    coronary_radius = max(float(np.linalg.norm(coronary_max - coronary_center)), 1.0)
    extra_poses = make_poses(
        coronary_center,
        max(radius, coronary_radius),
        args.extra_coronary_views,
        args.seed + 1000,
    )

    manifest = {
        "generator": "generate_heart_teachers_v2.py",
        "camera": {
            "projection": "perspective",
            "view_angle_degrees": args.view_angle,
            "base_views": args.views,
            "extra_coronary_views": args.extra_coronary_views,
            "width": args.width,
            "height": args.height,
            "center": center.tolist(),
            "radius": radius,
        },
        "split": {
            "files": ["train.txt", "test.txt"],
            "base_test_stride": 8,
            "base_test_views": 15,
            "extra_coronary_views_are_train_only": True,
        },
        "layers": [],
    }
    for index, (filename, name, color) in enumerate(LAYER_SPECS):
        output = os.path.join(args.output_dir, f"L{index}")
        view_count, sampled_count = render_layer(
            output,
            individual_surfaces,
            base_poses,
            extra_poses,
            args.width,
            args.height,
            args.view_angle,
            index,
            args.point_count,
        )
        entry = {
            "index": index,
            "name": name,
            "mask": filename,
            "color_rgb_0_1": list(color),
            "color_rgb_8bit": np.round(np.asarray(color) * 255).astype(int).tolist(),
            "views": view_count,
            "initial_points": sampled_count,
        }
        manifest["layers"].append(entry)
        print(
            f"Generated L{index} {name}: {view_count} views, "
            f"{sampled_count} current-layer points, color {entry['color_rgb_8bit']}"
        )

    with open(os.path.join(args.output_dir, "manifest.json"), "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)


if __name__ == "__main__":
    main()
