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


def read_mask_as_vtk(path):
    image = sitk.ReadImage(path)
    array = sitk.GetArrayFromImage(image).astype(np.uint8)
    vtk_image = vtk.vtkImageData()
    vtk_image.SetDimensions(image.GetSize())
    vtk_image.SetSpacing(image.GetSpacing())
    vtk_image.SetOrigin(image.GetOrigin())
    vtk_image.GetPointData().SetScalars(
        numpy_to_vtk(array.ravel(order="C"), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    )
    return vtk_image


def extract_surface(vtk_image):
    marching = vtk.vtkMarchingCubes()
    marching.SetInputData(vtk_image)
    marching.SetValue(0, 0.5)
    marching.ComputeNormalsOn()
    marching.Update()

    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputConnection(marching.GetOutputPort())
    smoother.SetNumberOfIterations(8)
    smoother.SetPassBand(0.12)
    smoother.BoundarySmoothingOff()
    smoother.FeatureEdgeSmoothingOff()
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.Update()
    return smoother.GetOutput()


def make_camera_poses(surface, count):
    bounds = surface.GetBounds()
    minimum = np.array(bounds[::2], dtype=float)
    maximum = np.array(bounds[1::2], dtype=float)
    center = (minimum + maximum) * 0.5
    radius = max(np.linalg.norm(maximum - center), 1.0)
    distance = radius * 2.4
    directions = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
    ]
    return [(center + directions[index] * distance, center, radius) for index in range(count)]


def camera_rotation(position, target):
    forward = target - position
    forward /= np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(forward, world_up)) > 0.95:
        world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    return np.vstack((right, down, forward))


def render_view(renderer, window, camera, position, target, radius, width, height, path):
    camera.SetPosition(*position)
    camera.SetFocalPoint(*target)
    camera.SetViewUp(0.0, 0.0, 1.0)
    camera.SetParallelProjection(True)
    camera.SetParallelScale(radius * 2.2)
    renderer.ResetCameraClippingRange()
    window.Render()

    capture = vtk.vtkWindowToImageFilter()
    capture.SetInput(window)
    capture.SetInputBufferTypeToRGBA()
    capture.ReadFrontBufferOff()
    capture.Update()
    vtk_image = capture.GetOutput()
    pixels = vtk_to_numpy(vtk_image.GetPointData().GetScalars())
    pixels = pixels.reshape((height, width, 4))[::-1]
    Image.fromarray(pixels, mode="RGBA").save(path)


def write_colmap(output, positions, targets, width, height, focal_length, points):
    sparse = os.path.join(output, "sparse", "0")
    os.makedirs(sparse, exist_ok=True)
    camera = colmap.Camera(
        id=1,
        model="SIMPLE_PINHOLE",
        width=width,
        height=height,
        params=np.array([focal_length, width / 2.0, height / 2.0]),
    )
    cameras = {1: camera}
    images = {}
    for index, (position, target) in enumerate(zip(positions, targets), start=1):
        rotation = camera_rotation(position, target)
        images[index] = colmap.Image(
            id=index,
            qvec=colmap.rotmat2qvec(rotation),
            tvec=-rotation @ position,
            camera_id=1,
            name=f"view_{index - 1:06}.png",
            xys=np.empty((0, 2)),
            point3D_ids=np.empty(0, dtype=np.int64),
        )
    point_records = {
        index: colmap.Point3D(
            id=index,
            xyz=point,
            rgb=np.array([220, 80, 80], dtype=np.uint8),
            error=0.0,
            image_ids=np.empty(0, dtype=np.int32),
            point2D_idxs=np.empty(0, dtype=np.int32),
        )
        for index, point in enumerate(points, start=1)
    }
    colmap.write_model(cameras, images, point_records, sparse, ext=".txt")


def main():
    parser = argparse.ArgumentParser(description="Generate a small heart teacher-view smoke test.")
    parser.add_argument("--mask", required=True, help="Binary NIfTI mask.")
    parser.add_argument("--output", required=True, help="Teacher output directory.")
    parser.add_argument("--views", type=int, default=2, choices=[2])
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    vtk_image = read_mask_as_vtk(args.mask)
    surface = extract_surface(vtk_image)
    if surface.GetNumberOfPoints() == 0:
        raise RuntimeError(f"Marching Cubes produced no surface for {args.mask}")

    os.makedirs(os.path.join(args.output, "images"), exist_ok=True)
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0.0, 0.0, 0.0)
    renderer.SetUseDepthPeeling(True)
    renderer.SetBackgroundAlpha(0.0)
    actor = vtk.vtkActor()
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(surface)
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0.86, 0.18, 0.18)
    actor.GetProperty().SetOpacity(1.0)
    renderer.AddActor(actor)

    window = vtk.vtkRenderWindow()
    window.SetOffScreenRendering(1)
    window.SetAlphaBitPlanes(1)
    window.SetSize(args.width, args.height)
    window.AddRenderer(renderer)
    camera = renderer.GetActiveCamera()
    poses = make_camera_poses(surface, args.views)
    positions = [pose[0] for pose in poses]
    targets = [pose[1] for pose in poses]
    radius = poses[0][2]
    for index, (position, target, _) in enumerate(poses):
        render_view(
            renderer,
            window,
            camera,
            position,
            target,
            radius,
            args.width,
            args.height,
            os.path.join(args.output, "images", f"view_{index:06}.png"),
        )

    points = vtk_to_numpy(surface.GetPoints().GetData())[:: max(1, surface.GetNumberOfPoints() // 60000)]
    focal_length = 0.5 * args.width / np.tan(np.deg2rad(30.0))
    write_colmap(args.output, positions, targets, args.width, args.height, focal_length, points)
    window.Finalize()
    print(f"Generated {args.views} views and {len(points)} surface points in {args.output}")


if __name__ == "__main__":
    main()