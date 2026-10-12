import argparse
import json
import os
import sys
import warnings

import numpy as np
import SimpleITK as sitk
from PIL import Image, ImageDraw
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy


sys.path.append(os.path.dirname(__file__))
import generate_heart_teachers_v2 as legacy


LAYER_FILENAMES = [spec[0] for spec in legacy.LAYER_SPECS]


def read_bool(path):
    image = sitk.ReadImage(path)
    return image, sitk.GetArrayFromImage(image) > 0


def crop_roi(image, start_zyx, stop_zyx):
    index = [int(start_zyx[2]), int(start_zyx[1]), int(start_zyx[0])]
    size = [int(stop_zyx[2] - start_zyx[2]),
            int(stop_zyx[1] - start_zyx[1]),
            int(stop_zyx[0] - start_zyx[0])]
    return sitk.RegionOfInterest(image, size=size, index=index)


def resample_isotropic(image, spacing, interpolator, default_value, pixel_id):
    old_size = np.asarray(image.GetSize(), dtype=np.int64)
    old_spacing = np.asarray(image.GetSpacing(), dtype=np.float64)
    new_spacing = np.full(3, spacing, dtype=np.float64)
    physical_extent = np.maximum(old_size - 1, 1) * old_spacing
    new_size = np.maximum(np.rint(physical_extent / new_spacing).astype(int) + 1, 2)
    return sitk.Resample(
        image,
        [int(v) for v in new_size],
        sitk.Transform(3, sitk.sitkIdentity),
        interpolator,
        image.GetOrigin(),
        [float(v) for v in new_spacing],
        image.GetDirection(),
        default_value,
        pixel_id,
    )


def vtk_image_from_sitk(image):
    array = sitk.GetArrayFromImage(image)
    vtk_image = vtk.vtkImageData()
    vtk_image.SetDimensions(image.GetSize())
    vtk_image.SetSpacing(image.GetSpacing())
    vtk_image.SetOrigin(image.GetOrigin())
    vtk_image.SetDirectionMatrix(image.GetDirection())
    if array.dtype == np.float32:
        vtk_type = vtk.VTK_FLOAT
    elif array.dtype == np.float64:
        vtk_type = vtk.VTK_DOUBLE
    else:
        vtk_type = vtk.VTK_UNSIGNED_CHAR
        array = array.astype(np.uint8)
    vtk_image.GetPointData().SetScalars(
        numpy_to_vtk(array.ravel(order="C"), deep=True, array_type=vtk_type)
    )
    return vtk_image


def sdf_surface(mask_image, sigma_mm, smooth_iterations=2, sdf_image=None):
    sdf = sdf_image if sdf_image is not None else sitk.SignedMaurerDistanceMap(
        sitk.Cast(mask_image, sitk.sitkUInt8),
        insideIsPositive=True,
        squaredDistance=False,
        useImageSpacing=True,
    )
    if sigma_mm > 0:
        sdf = sitk.SmoothingRecursiveGaussian(sdf, sigma_mm)
    contour = vtk.vtkFlyingEdges3D()
    contour.SetInputData(vtk_image_from_sitk(sitk.Cast(sdf, sitk.sitkFloat32)))
    contour.SetValue(0, 0.0)
    contour.ComputeNormalsOff()
    contour.Update()
    source = contour.GetOutputPort()
    if smooth_iterations > 0:
        smoother = vtk.vtkWindowedSincPolyDataFilter()
        smoother.SetInputConnection(source)
        smoother.SetNumberOfIterations(smooth_iterations)
        smoother.SetPassBand(0.18)
        smoother.BoundarySmoothingOff()
        smoother.FeatureEdgeSmoothingOff()
        smoother.NonManifoldSmoothingOn()
        smoother.NormalizeCoordinatesOn()
        smoother.Update()
        source = smoother.GetOutputPort()
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(source)
    normals.ComputePointNormalsOn()
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.SplittingOff()
    normals.Update()
    output = vtk.vtkPolyData()
    output.DeepCopy(normals.GetOutput())
    return output


def physical_to_index(points, image):
    origin = np.asarray(image.GetOrigin(), dtype=np.float64)
    spacing = np.asarray(image.GetSpacing(), dtype=np.float64)
    direction = np.asarray(image.GetDirection(), dtype=np.float64).reshape(3, 3)
    local = (points - origin) @ np.linalg.inv(direction).T
    return local / spacing


def sample_array(array_zyx, index_xyz, linear=True):
    xyz = np.asarray(index_xyz, dtype=np.float64)
    maximum = np.asarray(array_zyx.shape[::-1], dtype=np.int64) - 1
    xyz = np.clip(xyz, 0, maximum)
    if not linear:
        index = np.rint(xyz).astype(np.int64)
        return array_zyx[index[:, 2], index[:, 1], index[:, 0]]
    lo = np.floor(xyz).astype(np.int64)
    hi = np.minimum(lo + 1, maximum)
    fraction = xyz - lo
    x0, y0, z0 = lo[:, 0], lo[:, 1], lo[:, 2]
    x1, y1, z1 = hi[:, 0], hi[:, 1], hi[:, 2]
    wx, wy, wz = fraction[:, 0], fraction[:, 1], fraction[:, 2]
    c000 = array_zyx[z0, y0, x0]
    c100 = array_zyx[z0, y0, x1]
    c010 = array_zyx[z0, y1, x0]
    c110 = array_zyx[z0, y1, x1]
    c001 = array_zyx[z1, y0, x0]
    c101 = array_zyx[z1, y0, x1]
    c011 = array_zyx[z1, y1, x0]
    c111 = array_zyx[z1, y1, x1]
    c00 = c000 * (1 - wx) + c100 * wx
    c10 = c010 * (1 - wx) + c110 * wx
    c01 = c001 * (1 - wx) + c101 * wx
    c11 = c011 * (1 - wx) + c111 * wx
    c0 = c00 * (1 - wy) + c10 * wy
    c1 = c01 * (1 - wy) + c11 * wy
    return c0 * (1 - wz) + c1 * wz


def inward_median_hu(surface, ct_image, ct_array, mask_array):
    points = vtk_to_numpy(surface.GetPoints().GetData()).astype(np.float64)
    normals = vtk_to_numpy(surface.GetPointData().GetNormals()).astype(np.float64)
    samples = []
    for distance in (0.35, 0.70, 1.05, 1.40, 1.75, 2.10):
        minus = points - normals * distance
        plus = points + normals * distance
        minus_index = physical_to_index(minus, ct_image)
        plus_index = physical_to_index(plus, ct_image)
        minus_inside = sample_array(mask_array, minus_index, linear=False) > 0
        plus_inside = sample_array(mask_array, plus_index, linear=False) > 0
        minus_hu = sample_array(ct_array, minus_index, linear=True)
        plus_hu = sample_array(ct_array, plus_index, linear=True)
        samples.append(np.where(minus_inside, minus_hu, np.nan))
        samples.append(np.where(plus_inside, plus_hu, np.nan))
    values = np.stack(samples)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        hu = np.nanmedian(values, axis=0)
    missing = ~np.isfinite(hu)
    if np.any(missing):
        hu[missing] = sample_array(
            ct_array, physical_to_index(points[missing], ct_image), linear=True
        )
    return hu.astype(np.float32)


def heart_rgb(hu, points):
    intensity = np.clip((hu - 10.0) / 300.0, 0.0, 1.0) ** 0.72
    dark = np.array([96.0, 17.0, 19.0])
    light = np.array([202.0, 84.0, 62.0])
    rgb = dark[None, :] * (1 - intensity[:, None]) + light[None, :] * intensity[:, None]
    # Small deterministic variation prevents a plastic single-color appearance.
    texture = (np.sin(points[:, 0] * 0.31) + np.sin(points[:, 1] * 0.23)
               + np.sin(points[:, 2] * 0.27)) / 3.0
    rgb *= (0.97 + 0.03 * texture[:, None])
    return np.clip(rgb, 0, 255).astype(np.uint8)


def fat_rgb(hu, points):
    intensity = np.clip((hu + 190.0) / 160.0, 0.0, 1.0)
    dark = np.array([174.0, 123.0, 67.0])
    light = np.array([239.0, 207.0, 141.0])
    rgb = dark[None, :] * (1 - intensity[:, None]) + light[None, :] * intensity[:, None]
    return np.clip(rgb, 0, 255).astype(np.uint8)


def direct_color_actor(surface, rgb, roughness, specular, opacity=1.0):
    colored_surface = vtk.vtkPolyData()
    colored_surface.DeepCopy(surface)
    colors = numpy_to_vtk(rgb, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    colors.SetName("rgb")
    colors.SetNumberOfComponents(3)
    colored_surface.GetPointData().SetScalars(colors)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(colored_surface)
    mapper.SetColorModeToDirectScalars()
    mapper.SetScalarModeToUsePointData()
    mapper.ScalarVisibilityOn()
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    prop = actor.GetProperty()
    prop.SetInterpolationToPBR()
    prop.SetMetallic(0.0)
    prop.SetRoughness(roughness)
    prop.SetSpecular(specular)
    prop.SetOpacity(opacity)
    return actor


def material_actor(surface, color, roughness, specular, opacity=1.0):
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(surface)
    mapper.ScalarVisibilityOff()
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    prop = actor.GetProperty()
    prop.SetColor(*color)
    prop.SetInterpolationToPBR()
    prop.SetMetallic(0.0)
    prop.SetRoughness(roughness)
    prop.SetSpecular(specular)
    prop.SetOpacity(opacity)
    return actor


def add_lights(renderer, center, radius):
    renderer.AutomaticLightCreationOff()
    lights = [
        ((1.2, -1.0, 1.4), (1.0, 0.86, 0.78), 1.55),
        ((-1.3, -0.3, 0.5), (0.66, 0.76, 1.0), 0.70),
        ((0.2, 1.4, 1.0), (1.0, 0.55, 0.42), 0.82),
    ]
    for direction, color, intensity in lights:
        light = vtk.vtkLight()
        light.SetLightTypeToSceneLight()
        light.SetPosition(*(center + np.asarray(direction) * radius * 2.2))
        light.SetFocalPoint(*center)
        light.SetColor(*color)
        light.SetIntensity(intensity)
        renderer.AddLight(light)


def render_supersampled(path, actors, pose, center, radius, render_size, output_size):
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0.0, 0.0, 0.0)
    renderer.SetBackgroundAlpha(0.0)
    add_lights(renderer, center, radius)
    window = vtk.vtkRenderWindow()
    window.SetOffScreenRendering(1)
    window.SetAlphaBitPlanes(1)
    window.SetMultiSamples(8)
    window.SetSize(render_size, render_size)
    window.AddRenderer(renderer)
    for actor in actors:
        renderer.AddActor(actor)
    high_path = path + ".supersampled.png"
    legacy.render_png(
        renderer, window, renderer.GetActiveCamera(), pose[0], pose[1],
        60.0, high_path,
    )
    high = Image.open(high_path).convert("RGBA")
    high.resize((output_size, output_size), Image.Resampling.LANCZOS).save(path)
    os.remove(high_path)
    renderer.RemoveAllViewProps()
    window.Finalize()


def alpha_bbox(images, padding=14):
    alpha = np.maximum.reduce([np.asarray(image)[..., 3] for image in images])
    y, x = np.where(alpha > 0)
    return (max(int(x.min()) - padding, 0), max(int(y.min()) - padding, 0),
            min(int(x.max()) + padding + 1, images[0].width),
            min(int(y.max()) + padding + 1, images[0].height))


def write_comparison(paths, labels, output):
    images = [Image.open(path).convert("RGBA") for path in paths]
    target_size = (max(image.width for image in images),
                   max(image.height for image in images))
    images = [image if image.size == target_size else
              image.resize(target_size, Image.Resampling.LANCZOS)
              for image in images]
    box = alpha_bbox(images)
    tile, header = 480, 44
    canvas = Image.new("RGB", (tile * len(images), tile + header), (16, 18, 22))
    draw = ImageDraw.Draw(canvas)
    for i, (image, label) in enumerate(zip(images, labels)):
        crop = image.crop(box)
        background = Image.new("RGBA", crop.size, (24, 25, 28, 255))
        background.alpha_composite(crop)
        rgb = background.convert("RGB")
        rgb.thumbnail((tile - 12, tile - 12), Image.Resampling.LANCZOS)
        canvas.paste(rgb, (i * tile + (tile - rgb.width) // 2,
                           header + (tile - rgb.height) // 2))
        draw.text((i * tile + 12, 14), label, fill=(240, 240, 240))
    canvas.save(output)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a CT-informed high-quality exterior-heart teacher preview."
    )
    parser.add_argument("--ct", required=True)
    parser.add_argument("--heart-mask", required=True)
    parser.add_argument("--layers-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--original-teacher", default=None)
    parser.add_argument("--view-index", type=int, default=20)
    parser.add_argument("--isotropic-spacing", type=float, default=0.31)
    parser.add_argument("--sdf-sigma", type=float, default=1.00)
    parser.add_argument("--render-size", type=int, default=1024)
    parser.add_argument("--output-size", type=int, default=512)
    parser.add_argument("--fat-shell-mm", type=float, default=2.5)
    parser.add_argument("--fat-hu-low", type=float, default=-190.0)
    parser.add_argument("--fat-hu-high", type=float, default=-30.0)
    parser.add_argument("--minimum-fat-component", type=int, default=150)
    parser.add_argument("--fat-mask", default=None,
                        help="Optional prefiltered CT-grid fat mask for a preview")
    parser.add_argument("--fat-sdf-sigma", type=float, default=0.22)
    parser.add_argument("--fat-mesh-iterations", type=int, default=1)
    parser.add_argument("--fat-native-sdf", action="store_true",
                        help="Compute signed distance on native CT grid before isotropic resampling")
    parser.add_argument("--no-fat-cue", action="store_true",
                        help="Show only the separate fat mesh, without repainting the heart surface")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    ct_full = sitk.ReadImage(args.ct)
    heart_full = sitk.ReadImage(args.heart_mask)
    heart_array_full = sitk.GetArrayFromImage(heart_full) > 0
    points = np.argwhere(heart_array_full)
    spacing_zyx = np.asarray(ct_full.GetSpacing()[::-1])
    margin_zyx = np.ceil(10.0 / spacing_zyx).astype(int)
    start = np.maximum(points.min(0) - margin_zyx, 0)
    stop = np.minimum(points.max(0) + margin_zyx + 1, heart_array_full.shape)
    ct_crop = crop_roi(ct_full, start, stop)
    heart_crop = crop_roi(heart_full, start, stop)
    ct_iso = resample_isotropic(
        sitk.Cast(ct_crop, sitk.sitkFloat32), args.isotropic_spacing,
        sitk.sitkLinear, -1024.0, sitk.sitkFloat32,
    )
    heart_iso = resample_isotropic(
        sitk.Cast(heart_crop > 0, sitk.sitkUInt8), args.isotropic_spacing,
        sitk.sitkNearestNeighbor, 0.0, sitk.sitkUInt8,
    )
    ct = sitk.GetArrayFromImage(ct_iso).astype(np.float32)
    ct_material = sitk.GetArrayFromImage(
        sitk.SmoothingRecursiveGaussian(ct_iso, 0.60)
    ).astype(np.float32)
    heart = sitk.GetArrayFromImage(heart_iso) > 0

    heart_surface = sdf_surface(heart_iso, args.sdf_sigma, smooth_iterations=2)
    heart_points = vtk_to_numpy(heart_surface.GetPoints().GetData()).astype(np.float64)
    heart_hu = inward_median_hu(heart_surface, ct_iso, ct_material, heart)
    base_heart_rgb = heart_rgb(heart_hu, heart_points)
    exterior_actor = direct_color_actor(
        heart_surface, base_heart_rgb, roughness=0.54, specular=0.38
    )

    # Resample the coronary mask into the exact exterior ROI/reference grid.
    coronary_full = sitk.ReadImage(os.path.join(args.layers_dir, LAYER_FILENAMES[6]))
    coronary_crop = crop_roi(coronary_full, start, stop)
    coronary_iso = sitk.Resample(
        coronary_crop, heart_iso, sitk.Transform(3, sitk.sitkIdentity),
        sitk.sitkNearestNeighbor, 0.0, sitk.sitkUInt8,
    )
    coronary_surface = sdf_surface(coronary_iso, 0.16, smooth_iterations=1)
    coronary_distance_image = sitk.SignedMaurerDistanceMap(
        coronary_iso, insideIsPositive=False, squaredDistance=False,
        useImageSpacing=True,
    )
    coronary_distance = sitk.GetArrayFromImage(coronary_distance_image).astype(np.float32)
    surface_indices = physical_to_index(heart_points, ct_iso)
    coronary_distance_surface = sample_array(coronary_distance, surface_indices, linear=True)
    coronary_weight = np.clip((2.2 - coronary_distance_surface) / 1.5, 0.0, 1.0)
    coronary_color = np.array([56.0, 3.0, 5.0])
    coronary_rgb = (
        base_heart_rgb.astype(np.float32) * (1.0 - 0.92 * coronary_weight[:, None])
        + coronary_color[None, :] * (0.92 * coronary_weight[:, None])
    ).astype(np.uint8)
    coronary_painted_actor = direct_color_actor(
        heart_surface, coronary_rgb, roughness=0.52, specular=0.42
    )
    del coronary_distance, coronary_distance_image

    # Conservative fat candidate; it remains a separate optional teacher variant.
    distance = sitk.GetArrayFromImage(
        sitk.SignedMaurerDistanceMap(
            heart_iso, insideIsPositive=False, squaredDistance=False,
            useImageSpacing=True,
        )
    )
    fat_sdf_iso = None
    if args.fat_mask:
        fat_full = sitk.ReadImage(args.fat_mask)
        fat_crop = crop_roi(fat_full, start, stop)
        if args.fat_native_sdf:
            native_sdf = sitk.SignedMaurerDistanceMap(
                sitk.Cast(fat_crop > 0, sitk.sitkUInt8),
                insideIsPositive=True, squaredDistance=False,
                useImageSpacing=True,
            ) + min(fat_crop.GetSpacing()) * 0.5
            fat_sdf_iso = sitk.Resample(
                native_sdf, heart_iso, sitk.Transform(3, sitk.sitkIdentity),
                sitk.sitkLinear, -12.0, sitk.sitkFloat32,
            )
        fat_iso = sitk.Resample(
            fat_crop, heart_iso, sitk.Transform(3, sitk.sitkIdentity),
            sitk.sitkNearestNeighbor, 0.0, sitk.sitkUInt8,
        )
        fat = sitk.GetArrayFromImage(fat_iso) > 0
    else:
        fat = ((~heart) & (distance > 0) & (distance <= args.fat_shell_mm)
               & (ct >= args.fat_hu_low) & (ct <= args.fat_hu_high))
    components = sitk.ConnectedComponent(sitk.GetImageFromArray(fat.astype(np.uint8)))
    components = sitk.RelabelComponent(
        components, minimumObjectSize=args.minimum_fat_component
    )
    fat = sitk.GetArrayFromImage(components) > 0
    fat_image = sitk.GetImageFromArray(fat.astype(np.uint8))
    fat_image.CopyInformation(heart_iso)
    fat_surface = sdf_surface(
        fat_image, args.fat_sdf_sigma,
        smooth_iterations=args.fat_mesh_iterations,
        sdf_image=fat_sdf_iso,
    )
    fat_points = vtk_to_numpy(fat_surface.GetPoints().GetData()).astype(np.float64)
    fat_hu = inward_median_hu(fat_surface, ct_iso, ct, fat)
    fat_actor = direct_color_actor(
        fat_surface, fat_rgb(fat_hu, fat_points), roughness=0.66,
        specular=0.26, opacity=0.96,
    )
    fat_distance_image = sitk.SignedMaurerDistanceMap(
        fat_image, insideIsPositive=False, squaredDistance=False,
        useImageSpacing=True,
    )
    fat_distance = sitk.GetArrayFromImage(fat_distance_image).astype(np.float32)
    fat_distance_surface = sample_array(fat_distance, surface_indices, linear=True)
    fat_weight = np.clip((1.15 - fat_distance_surface) / 0.85, 0.0, 1.0)
    fat_surface_color = np.array([205.0, 136.0, 42.0])
    fat_cue_rgb = (
        coronary_rgb.astype(np.float32) * (1.0 - 0.82 * fat_weight[:, None])
        + fat_surface_color[None, :] * (0.82 * fat_weight[:, None])
    ).astype(np.uint8)
    fat_cue_actor = direct_color_actor(
        heart_surface, fat_cue_rgb, roughness=0.62, specular=0.25
    )
    del fat_distance, fat_distance_image

    # Use the exact existing teacher camera distribution.
    layer_arrays = []
    reference = None
    for filename in LAYER_FILENAMES:
        image, array = read_bool(os.path.join(args.layers_dir, filename))
        reference = image if reference is None else reference
        layer_arrays.append(array)
    union_surface = legacy.extract_surface(reference, np.logical_or.reduce(layer_arrays))
    minimum, maximum = legacy.surface_bounds(union_surface)
    center = (minimum + maximum) * 0.5
    radius = max(float(np.linalg.norm(maximum - center)), 1.0)
    pose = legacy.make_poses(center, radius, 120, 42)[args.view_index]

    exterior_path = os.path.join(args.output_dir, f"view_{args.view_index:06}_exterior.png")
    coronary_path = os.path.join(
        args.output_dir, f"view_{args.view_index:06}_exterior_coronaries.png"
    )
    fat_path = os.path.join(
        args.output_dir, f"view_{args.view_index:06}_exterior_coronaries_fat.png"
    )
    render_supersampled(
        exterior_path, [exterior_actor], pose, center, radius,
        args.render_size, args.output_size,
    )
    render_supersampled(
        coronary_path, [coronary_painted_actor], pose, center, radius,
        args.render_size, args.output_size,
    )
    render_supersampled(
        fat_path,
        [coronary_painted_actor, fat_actor] if args.no_fat_cue else
        [fat_cue_actor, fat_actor],
        pose, center, radius,
        args.render_size, args.output_size,
    )
    fat_only_path = os.path.join(
        args.output_dir, f"view_{args.view_index:06}_fat_only.png"
    )
    render_supersampled(
        fat_only_path, [fat_actor], pose, center, radius,
        args.render_size, args.output_size,
    )

    comparison = None
    if args.original_teacher:
        comparison = os.path.join(args.output_dir, "exterior-teacher-v4-comparison.png")
        fat_label = ("V4 exterior + prefiltered fat mask (diagnostic)"
                     if args.fat_mask else
                     "V4 exterior + 2.5 mm fat candidate (diagnostic)")
        write_comparison(
            [args.original_teacher, coronary_path, fat_path],
            ["Current seven-layer teacher", "V4 exterior + dark-red coronaries",
             fat_label],
            comparison,
        )

    sitk.WriteImage(heart_iso, os.path.join(args.output_dir, "heart_isotropic_031mm.nii.gz"), True)
    sitk.WriteImage(fat_image, os.path.join(args.output_dir, "fat_candidate_25mm.nii.gz"), True)
    voxel_ml = args.isotropic_spacing ** 3 / 1000.0
    summary = {
        "view_index": args.view_index,
        "isotropic_spacing_mm": args.isotropic_spacing,
        "sdf_sigma_mm": args.sdf_sigma,
        "render_size": args.render_size,
        "output_size": args.output_size,
        "isotropic_volume_size_xyz": list(map(int, heart_iso.GetSize())),
        "heart_surface_points": int(heart_surface.GetNumberOfPoints()),
        "coronary_surface_points": int(coronary_surface.GetNumberOfPoints()),
        "fat_surface_points": int(fat_surface.GetNumberOfPoints()),
        "fat_candidate_ml": float(fat.sum() * voxel_ml),
        "fat_mask_source": args.fat_mask,
        "fat_sdf_sigma_mm": args.fat_sdf_sigma,
        "fat_mesh_iterations": args.fat_mesh_iterations,
        "fat_native_sdf": args.fat_native_sdf,
        "fat_cue_painted_on_heart": not args.no_fat_cue,
        "fat_only_preview": fat_only_path,
        "heart_hu_percentiles": np.percentile(
            heart_hu, [1, 5, 25, 50, 75, 95, 99]
        ).round(3).tolist(),
        "comparison": comparison,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
