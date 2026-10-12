"""Volume-preserving surface regularization of the full heart-adjacent fat ROI.

Keeps the source segmentation as evidence and stores all derived outputs
separately. The source is a torso-fat ROI, not a dedicated epicardial label.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from PIL import Image, ImageDraw
import vtk

import generate_heart_exterior_teachers_v4 as exterior
import generate_heart_teachers_v2 as legacy
import generate_heart_teachers_v5_sdf as surfaces


def field_image(array, reference):
    image = sitk.GetImageFromArray(array)
    image.CopyInformation(reference)
    return image


def signed_distance(mask):
    return sitk.SignedMaurerDistanceMap(
        sitk.Cast(mask > 0, sitk.sitkUInt8), insideIsPositive=True,
        squaredDistance=False, useImageSpacing=True,
    ) + min(mask.GetSpacing()) * 0.5


def resample(image, reference, interpolation=sitk.sitkLinear, default=-20):
    return sitk.Resample(image, reference, sitk.Transform(3, sitk.sitkIdentity),
                         interpolation, default, sitk.sitkFloat32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='.',
                        help='Project root containing output, masks, and NIfTI (default: current directory)')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--sigma-mm', type=float, default=0.6)
    parser.add_argument('--heart-sigma-mm', type=float, default=1.0)
    args = parser.parse_args()
    root, output = Path(args.root), Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    source_path = root / 'output/L7_tissue_types_wsl/review/heart_shell_diagnostic.nii.gz'
    source = sitk.ReadImage(str(source_path))
    heart = sitk.ReadImage(str(root / 'masks/total_free/heart.nii.gz'))
    ct = sitk.ReadImage(str(root / 'NIfTI/DSCorAdSeq075B26fBestDiast75_7.nii.gz'))
    geometry = lambda i: (i.GetSize(), i.GetSpacing(), i.GetOrigin(), i.GetDirection())
    if geometry(source) != geometry(ct) or geometry(heart) != geometry(ct):
        raise ValueError('CT/mask geometry mismatch')
    source_array = sitk.GetArrayFromImage(source) > 0
    source_ml = float(source_array.sum() * np.prod(source.GetSpacing()) / 1000)
    coords = np.where(sitk.GetArrayFromImage(heart) > 0)
    spacing = np.asarray(source.GetSpacing()[::-1])
    pad = np.ceil(10 / spacing).astype(int)
    lo = np.maximum(np.array([c.min() for c in coords]) - pad, 0)
    hi = np.minimum(np.array([c.max() for c in coords]) + pad + 1, source_array.shape)
    source_roi = exterior.crop_roi(source, lo, hi)
    heart_roi = exterior.crop_roi(heart, lo, hi)
    ct_roi = exterior.crop_roi(ct, lo, hi)
    source_roi_array = sitk.GetArrayFromImage(source_roi) > 0

    # Smooth the occupancy on the native grid in physical units. Interpolate
    # the continuous field, never an upsampled binary staircase.
    probability_image = sitk.SmoothingRecursiveGaussian(
        sitk.Cast(source_roi > 0, sitk.sitkFloat32), args.sigma_mm)
    probability_iso = exterior.resample_isotropic(
        probability_image, 0.4, sitk.sitkLinear, 0, sitk.sitkFloat32)
    p = sitk.GetArrayFromImage(probability_iso)
    heart_sdf = sitk.SmoothingRecursiveGaussian(signed_distance(heart_roi), args.heart_sigma_mm)
    heart_sdf_iso = resample(heart_sdf, probability_iso)
    h = sitk.GetArrayFromImage(heart_sdf_iso)
    source_sdf_iso = resample(signed_distance(source_roi), probability_iso)
    source_sdf = sitk.GetArrayFromImage(source_sdf_iso)
    # Small hidden overlap makes contact robust to heart-surface smoothing.
    # Keep every relocated surface within 1 mm of the CT-supported fat.
    allowed = (h <= 0.20) & (source_sdf >= -1.0) & (p > 0.005)
    voxel_ml = float(np.prod(probability_iso.GetSpacing()) / 1000)
    target_count = int(round(source_ml / voxel_ml))
    values = p[allowed]
    if len(values) <= target_count:
        raise ValueError('Insufficient supported space for volume-preserving surface')
    boundary = len(values) - target_count
    partition = np.partition(values, (boundary - 1, boundary))
    threshold = float((partition[boundary - 1] + partition[boundary]) * 0.5)
    del values, partition
    implicit = np.minimum(p - threshold, (0.20 - h) / 1.0)
    implicit = np.minimum(implicit, (source_sdf + 1.0) / 1.0).astype(np.float32)
    field = field_image(implicit, probability_iso)
    mask = implicit > 0
    mask_image = field_image(mask.astype(np.uint8), probability_iso)
    # Data reach the superior CT boundary. Explicit outside padding closes
    # the clipped surface at that boundary without extrapolating tissue.
    mesh_field = sitk.ConstantPad(field, [1, 1, 1], [1, 1, 1], -1.0)
    sitk.WriteImage(mesh_field, str(output / 'L7_fat_surface_field_iso.nii.gz'), True)
    sitk.WriteImage(mask_image, str(output / 'L7_fat_preserved_iso.nii.gz'), True)
    native_mask = sitk.Resample(mask_image, source, sitk.Transform(3, sitk.sitkIdentity),
                               sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    sitk.WriteImage(native_mask, str(output / 'L7_fat_preserved_ct_grid.nii.gz'), True)
    new_native = sitk.GetArrayFromImage(native_mask) > 0
    original_hu = sitk.GetArrayFromImage(ct)
    overlap = int((new_native & source_array).sum())
    stats = {
        'source': str(source_path), 'source_volume_ml': source_ml,
        'surface_mask_volume_ml': float(mask.sum() * voxel_ml),
        'volume_change_percent': float(100 * (mask.sum() * voxel_ml / source_ml - 1)),
        'original_voxels_retained_percent': float(100 * overlap / source_array.sum()),
        'native_grid_dice': float(2 * overlap / (source_array.sum() + new_native.sum())),
        'original_fat_hu_percentiles': np.percentile(original_hu[source_array], [5, 50, 95]).tolist(),
        'regularized_hu_percentiles': np.percentile(original_hu[new_native], [5, 50, 95]).tolist(),
        'regularized_in_original_hu_band_percent': float(100 * np.mean(
            (original_hu[new_native] >= -190) & (original_hu[new_native] <= -30))),
        'sigma_mm': args.sigma_mm, 'occupancy_threshold': threshold,
        'heart_contact_sdf_sigma_mm': args.heart_sigma_mm,
        'isotropic_spacing_mm': 0.4, 'max_outward_distance_from_source_mm': 1.0,
        'hidden_overlap_in_heart_mm': 0.2,
        'note': 'Heart-adjacent torso-fat ROI; surface regularization, not new tissue detection.',
    }
    print(json.dumps(stats, indent=2), flush=True)

    fat_surface = surfaces.surface_from_sdf(mesh_field, 12, 0.12)
    writer = vtk.vtkPLYWriter()
    writer.SetFileName(str(output / 'L7_fat_surface.ply'))
    writer.SetInputData(fat_surface)
    writer.SetFileTypeToBinary()
    writer.Write()
    triangulate = vtk.vtkTriangleFilter()
    triangulate.SetInputData(fat_surface)
    mass = vtk.vtkMassProperties()
    mass.SetInputConnection(triangulate.GetOutputPort())
    mass.Update()
    stats['mesh_volume_ml'] = mass.GetVolume() / 1000
    edges = vtk.vtkFeatureEdges()
    edges.SetInputData(fat_surface)
    edges.BoundaryEdgesOn()
    edges.NonManifoldEdgesOn()
    edges.FeatureEdgesOff()
    edges.ManifoldEdgesOff()
    edges.Update()
    stats['mesh_open_or_nonmanifold_edges'] = edges.GetOutput().GetNumberOfCells()
    stats['scan_boundary_cap'] = 'One outside voxel pads the field; scan-clipped tissue is capped, not extrapolated.'

    heart_surface = surfaces.surface_from_sdf(
        sitk.ConstantPad(heart_sdf_iso, [1, 1, 1], [1, 1, 1], -20.0), 12, 0.10)
    heart_actor = exterior.material_actor(heart_surface, (0.56, 0.105, 0.085), 0.58, 0.30)
    # Identical, restrained material for source and result isolates the
    # geometric change. No HU stripe texture is painted on the heart.
    fat_color = (0.84, 0.70, 0.47)
    fat_actor = exterior.material_actor(fat_surface, fat_color, 0.58, 0.30)
    old_padded = sitk.ConstantPad(source_roi, [1, 1, 1], [1, 1, 1], 0)
    old_surface = legacy.extract_surface(old_padded, sitk.GetArrayFromImage(old_padded) > 0, iterations=12)
    old_actor = exterior.material_actor(old_surface, fat_color, 0.58, 0.30)
    cor_image = sitk.ReadImage(str(root / 'masks/layers/L6_coronary_arteries.nii.gz'))
    cor_roi = exterior.crop_roi(cor_image, lo, hi)
    cor_sdf = resample(signed_distance(cor_roi), probability_iso)
    cor_surface = surfaces.surface_from_sdf(sitk.ConstantPad(
        sitk.SmoothingRecursiveGaussian(cor_sdf, 0.16), [1, 1, 1], [1, 1, 1], -20.0), 12, 0.18)
    cor_actor = exterior.material_actor(cor_surface, (0.30, 0.025, 0.03), 0.50, 0.30)
    camera_config = json.loads((root / 'teachers_v4/manifest.json').read_text())['camera']
    center = np.asarray(camera_config['center'])
    radius = camera_config['radius']
    poses = legacy.make_poses(center, radius, 120, 42)
    montage_rows = []
    for view in (20, 70, 100):
        paths = []
        for label, actors in (
                ('source', [heart_actor, cor_actor, old_actor]),
                ('preserved', [heart_actor, cor_actor, fat_actor]),
                ('fat_only', [fat_actor])):
            path = output / f'view_{view:03}_{label}.png'
            exterior.render_supersampled(str(path), actors, poses[view], center, radius, 1024, 768)
            paths.append(str(path))
        row = output / f'comparison_{view:03}.png'
        exterior.write_comparison(paths, ['Source fat ROI', 'Volume-preserved smooth surface', 'Fat only'], str(row))
        montage_rows.append(Image.open(row).convert('RGB'))
        print(f'Rendered view {view}', flush=True)
    montage = Image.new('RGB', (montage_rows[0].width, sum(i.height for i in montage_rows)))
    y = 0
    for row in montage_rows:
        montage.paste(row, (0, y))
        y += row.height
    montage.save(output / 'multiview_comparison.png')
    # Native-grid slices: yellow retained source, cyan additions, red removals.
    panels = []
    for z in (67, 100, 144, 169, 189, 218):
        ys, xs = slice(40, 400), slice(55, 420)
        gray = np.clip((original_hu[z, ys, xs] + 150) / 650, 0, 1)
        rgb = np.repeat((gray * 255).astype(np.uint8)[..., None], 3, axis=2)
        a, b = source_array[z, ys, xs], new_native[z, ys, xs]
        rgb[a & b] = (235, 205, 70)
        rgb[b & ~a] = (45, 215, 240)
        rgb[a & ~b] = (245, 65, 65)
        panel = Image.new('RGB', (rgb.shape[1], rgb.shape[0] + 26), (15, 16, 18))
        panel.paste(Image.fromarray(rgb), (0, 26))
        ImageDraw.Draw(panel).text((6, 6), f'z={z}: retained / added / removed', fill='white')
        panels.append(panel)
    w, height = panels[0].size
    axial = Image.new('RGB', (w * 3, height * 2))
    for i, panel in enumerate(panels):
        axial.paste(panel, ((i % 3) * w, (i // 3) * height))
    axial.save(output / 'ct_boundary_changes.png')
    (output / 'metrics.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
