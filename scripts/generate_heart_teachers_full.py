import argparse
import os
import sys

import numpy as np
import SimpleITK as sitk
from PIL import Image
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

sys.path.append(os.path.dirname(__file__))
import read_write_model as colmap


LAYER_FILES = [
    "L0_left_ventricle.nii.gz",
    "L1_right_ventricle.nii.gz",
    "L2_left_atrium.nii.gz",
    "L3_right_atrium.nii.gz",
    "L4_myocardium.nii.gz",
    "L5_great_vessels.nii.gz",
    "L6_coronary_arteries.nii.gz",
]


def read_mask(path):
    image = sitk.ReadImage(path)
    return image, sitk.GetArrayFromImage(image).astype(np.uint8)


def mask_to_vtk(image, array):
    vtk_image = vtk.vtkImageData()
    vtk_image.SetDimensions(image.GetSize())
    vtk_image.SetSpacing(image.GetSpacing())
    vtk_image.SetOrigin(image.GetOrigin())
    vtk_image.GetPointData().SetScalars(
        numpy_to_vtk(array.ravel(order="C"), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
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
    center = (minimum + maximum) * 0.5
    radius = max(float(np.linalg.norm(maximum - center)), 1.0)
    return center, radius


def make_poses(center, radius, count, seed):
    rng = np.random.default_rng(seed)
    indices = np.arange(count, dtype=float) + 0.5
    z = 1.0 - 2.0 * indices / count
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    theta = golden_angle * indices
    directions = np.column_stack((np.sqrt(1.0 - z * z) * np.cos(theta),
                                  np.sqrt(1.0 - z * z) * np.sin(theta), z))
    directions += rng.normal(0.0, 0.002, directions.shape)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    positions = center + directions * (radius * 2.4)
    return [(position, center) for position in positions]


def set_camera(camera, position, target, radius):
    camera.SetPosition(*position)
    camera.SetFocalPoint(*target)
    camera.SetViewUp(0.0, 0.0, 1.0)
    camera.SetParallelProjection(True)
    camera.SetParallelScale(radius * 2.2)


def render_png(renderer, window, camera, position, target, radius, path):
    set_camera(camera, position, target, radius)
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


def camera_rotation(position, target):
    forward = target - position
    forward /= np.linalg.norm(forward)
    up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(forward, up)) > 0.95:
        up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    return np.vstack((right, down, forward))


def write_colmap(output, poses, width, height, focal_length, points):
    sparse = os.path.join(output, "sparse", "0")
    os.makedirs(sparse, exist_ok=True)
    cameras = {1: colmap.Camera(1, "SIMPLE_PINHOLE", width, height,
                                np.array([focal_length, width / 2.0, height / 2.0]))}
    images = {}
    for index, (position, target) in enumerate(poses, start=1):
        rotation = camera_rotation(position, target)
        images[index] = colmap.Image(
            index, colmap.rotmat2qvec(rotation), -rotation @ position, 1,
            f"view_{index - 1:06}.png", np.empty((0, 2)), np.empty(0, dtype=np.int64)
        )
    point_records = {
        index: colmap.Point3D(index, point, np.array([220, 80, 80], dtype=np.uint8),
                              0.0, np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32))
        for index, point in enumerate(points, start=1)
    }
    colmap.write_model(cameras, images, point_records, sparse, ext=".txt")


def sample_points(surface, count, seed):
    points = vtk_to_numpy(surface.GetPoints().GetData()).astype(np.float64)
    if len(points) <= count:
        return points
    rng = np.random.default_rng(seed)
    return points[rng.choice(len(points), count, replace=False)]


def build_actor(surface, color):
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(surface)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*color)
    actor.GetProperty().SetOpacity(1.0)
    return actor


def render_dataset(layer_output, surfaces, poses, radius, width, height, layer_index,
                   point_count, extra_coronary_views):
    images_dir = os.path.join(layer_output, "images")
    os.makedirs(images_dir, exist_ok=True)
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0.0, 0.0, 0.0)
    renderer.SetUseDepthPeeling(True)
    renderer.SetBackgroundAlpha(0.0)
    window = vtk.vtkRenderWindow()
    window.SetOffScreenRendering(1)
    window.SetAlphaBitPlanes(1)
    window.SetSize(width, height)
    window.AddRenderer(renderer)
    camera = renderer.GetActiveCamera()
    # Each surface is already the cumulative union through its layer.
    renderer.AddActor(build_actor(surfaces[layer_index], (0.86, 0.18, 0.18)))

    all_poses = list(poses)
    if layer_index == 6 and extra_coronary_views:
        coronary_center, coronary_radius = surface_bounds(surfaces[6])
        all_poses += make_poses(coronary_center, coronary_radius, extra_coronary_views, 1042)
    for index, (position, target) in enumerate(all_poses):
        render_png(renderer, window, camera, position, target, radius,
                   os.path.join(images_dir, f"view_{index:06}.png"))

    focal_length = 0.5 * width / np.tan(np.deg2rad(30.0))
    points = sample_points(surfaces[layer_index], point_count, 42 + layer_index)
    write_colmap(layer_output, all_poses, width, height, focal_length, points)
    train_count = 105 + (extra_coronary_views if layer_index == 6 else 0)
    with open(os.path.join(layer_output, "train.txt"), "w", encoding="ascii") as file:
        file.write("\n".join(f"view_{index:06}.png" for index in range(train_count)))
    with open(os.path.join(layer_output, "test.txt"), "w", encoding="ascii") as file:
        file.write("\n".join(f"view_{index:06}.png" for index in range(105, 120)))
    renderer.RemoveAllViewProps()
    window.Finalize()


def main():
    parser = argparse.ArgumentParser(description="Generate cumulative heart teacher views.")
    parser.add_argument("--layers-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--views", type=int, default=120)
    parser.add_argument("--extra-coronary-views", type=int, default=60)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--point-count", type=int, default=60000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.views != 120:
        raise ValueError("The formal split requires exactly 120 shared views")
    if args.views < 105:
        raise ValueError("views must be at least 105")

    reference, _ = read_mask(os.path.join(args.layers_dir, LAYER_FILES[0]))
    arrays = []
    for filename in LAYER_FILES:
        image, array = read_mask(os.path.join(args.layers_dir, filename))
        if image.GetSize() != reference.GetSize() or image.GetSpacing() != reference.GetSpacing():
            raise ValueError(f"Geometry mismatch in {filename}")
        arrays.append(array != 0)
    cumulative = np.zeros_like(arrays[0], dtype=bool)
    surfaces = []
    for index, array in enumerate(arrays):
        cumulative |= array
        surfaces.append(extract_surface(reference, cumulative.astype(np.uint8)))
        if surfaces[-1].GetNumberOfPoints() == 0:
            raise RuntimeError(f"No surface extracted for L{index}")

    center, radius = surface_bounds(surfaces[0])
    poses = make_poses(center, radius, args.views, args.seed)
    for index in range(7):
        output = os.path.join(args.output_dir, f"L{index}")
        render_dataset(output, surfaces, poses, radius, args.width, args.height, index,
                       args.point_count, args.extra_coronary_views)
        print(f"Generated L{index}: {120 + (args.extra_coronary_views if index == 6 else 0)} views")


if __name__ == "__main__":
    main()