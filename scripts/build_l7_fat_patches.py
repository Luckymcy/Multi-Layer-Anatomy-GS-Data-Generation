"""Generate conservative, CT-supported pericoronary fat candidates for review.

This is an anatomical hypothesis, not a clinical segmentation. It never
overwrites the TotalSegmentator output or any existing layer.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi


def read(path):
    image = sitk.ReadImage(str(path))
    return image, sitk.GetArrayFromImage(image)


def ellipsoid(radius_mm, spacing):
    extents = np.ceil(radius_mm / spacing).astype(int)
    grid = np.ogrid[tuple(slice(-n, n + 1) for n in extents)]
    distance2 = sum((axis * step) ** 2 for axis, step in zip(grid, spacing))
    return distance2 <= radius_mm ** 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ct", required=True)
    parser.add_argument("--torso-fat", required=True)
    parser.add_argument("--heart", required=True)
    parser.add_argument("--coronary", required=True)
    parser.add_argument("--layers-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ct_image, ct = read(args.ct)
    fat_image, fat = read(args.torso_fat)
    heart_image, heart = read(args.heart)
    coronary_image, coronary = read(args.coronary)
    geometry = lambda i: (i.GetSize(), i.GetSpacing(), i.GetOrigin(), i.GetDirection())
    if not all(geometry(i) == geometry(ct_image) for i in (fat_image, heart_image, coronary_image)):
        raise ValueError("Input geometry mismatch")
    fat, heart, coronary = fat > 0, heart > 0, coronary > 0
    coords = np.where(heart)
    spacing = np.asarray(ct_image.GetSpacing()[::-1])
    pad = np.ceil(12 / spacing).astype(int)
    lo = np.maximum(np.array([c.min() for c in coords]) - pad, 0)
    hi = np.minimum(np.array([c.max() for c in coords]) + pad + 1, ct.shape)
    box = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    h, f, c, hu = heart[box], fat[box], coronary[box], ct[box]
    occupied = np.zeros(h.shape, dtype=bool)
    for layer in sorted(Path(args.layers_dir).glob("L*.nii.gz")):
        image, array = read(layer)
        if geometry(image) != geometry(ct_image):
            raise ValueError(f"Layer geometry mismatch: {layer}")
        occupied |= array[box] > 0
    d_heart = ndi.distance_transform_edt(~h, sampling=spacing)
    d_coronary = ndi.distance_transform_edt(~c, sampling=spacing)
    ct_fat = (hu >= -190) & (hu <= -30)
    outer = f & ct_fat & ~h & (d_heart <= 4)
    inner = f & ct_fat & h & ~occupied
    voxel_ml = float(np.prod(ct_image.GetSpacing()) / 1000)
    metrics = {"source": "TotalSegmentator tissue_types/torso_fat + native CT HU",
               "voxel_ml": voxel_ml, "unfiltered_outer_ml": float(outer.sum() * voxel_ml),
               "unfiltered_inner_ml": float(inner.sum() * voxel_ml), "variants": {}}
    for name, coronary_mm, opening_mm in (("patches_4mm", 4.0, 0.7),
                                          ("patches_5mm", 5.0, 0.65),
                                          ("patches_6mm", 6.0, 0.6)):
        candidate = (outer | inner) & (d_coronary <= coronary_mm)
        before = int(candidate.sum())
        candidate = ndi.binary_opening(candidate, structure=ellipsoid(opening_mm, spacing))
        labels, n = ndi.label(candidate, structure=ndi.generate_binary_structure(3, 2))
        counts = np.bincount(labels.ravel())
        # Below about 0.03 mL these disconnected islands appear as floating
        # speckles at the teacher camera distance rather than coherent pads.
        minimum_component_voxels = int(np.ceil(0.03 / voxel_ml))
        keep = np.where(counts >= minimum_component_voxels)[0]
        keep = keep[keep != 0]
        candidate = np.isin(labels, keep)
        full = np.zeros(ct.shape, dtype=np.uint8)
        full[box] = candidate
        image = sitk.GetImageFromArray(full)
        image.CopyInformation(ct_image)
        sitk.WriteImage(image, str(out / f"{name}.nii.gz"), True)
        metrics["variants"][name] = {
            "near_coronary_before_opening_ml": before * voxel_ml,
            "retained_ml": float(candidate.sum() * voxel_ml),
            "retained_components": len(keep),
            "largest_component_ml": float(counts[keep].max() * voxel_ml) if len(keep) else 0,
            "hu_percentiles": np.percentile(hu[candidate], [5, 25, 50, 75, 95]).tolist() if candidate.any() else [],
            "coronary_distance_limit_mm": coronary_mm,
            "opening_radius_mm": opening_mm,
            "minimum_component_ml": 0.03,
        }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
