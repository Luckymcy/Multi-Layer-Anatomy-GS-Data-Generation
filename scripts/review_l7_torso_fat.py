"""Review TotalSegmentator torso-fat near the heart; diagnostic only."""

import argparse
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi
from PIL import Image, ImageDraw


def read(path):
    image = sitk.ReadImage(str(path))
    return image, sitk.GetArrayFromImage(image)


def check_grid(reference, other, name):
    if (reference.GetSize(), reference.GetSpacing(), reference.GetOrigin(), reference.GetDirection()) != (
        other.GetSize(), other.GetSpacing(), other.GetOrigin(), other.GetDirection()
    ):
        raise ValueError(f"{name}: grid differs from CT")


def overlay(ct, heart, coronary, fat, z):
    ys, xs = slice(40, 400), slice(55, 420)
    base = np.clip((ct[z, ys, xs] + 150) / 650, 0, 1)
    rgb = np.repeat((base * 255).astype(np.uint8)[..., None], 3, axis=2)
    outlines = heart[z, ys, xs] & ~ndi.binary_erosion(heart[z, ys, xs])
    rgb[outlines] = [70, 190, 110]
    rgb[coronary[z, ys, xs]] = [230, 55, 55]
    region = fat[z, ys, xs]
    rgb[region] = (0.20 * rgb[region] + 0.80 * np.array([255, 210, 45])).astype(np.uint8)
    panel = Image.new("RGB", (rgb.shape[1], rgb.shape[0] + 26), (20, 20, 20))
    panel.paste(Image.fromarray(rgb), (0, 26))
    ImageDraw.Draw(panel).text((5, 5), f"z={z}  fat={int(region.sum())}", fill="white")
    return panel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ct", required=True)
    parser.add_argument("--torso-fat", required=True)
    parser.add_argument("--heart", required=True)
    parser.add_argument("--coronary", required=True)
    parser.add_argument("--layers-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    ct_image, ct = read(args.ct)
    fat_image, torso_fat = read(args.torso_fat)
    heart_image, heart = read(args.heart)
    coronary_image, coronary = read(args.coronary)
    for name, image in (("torso_fat", fat_image), ("heart", heart_image), ("coronary", coronary_image)):
        check_grid(ct_image, image, name)
    torso_fat, heart, coronary = (x > 0 for x in (torso_fat, heart, coronary))

    coords = np.where(heart)
    spacing = np.asarray(ct_image.GetSpacing()[::-1])
    pad = np.ceil(12 / spacing).astype(int)
    low = np.maximum(np.array([x.min() for x in coords]) - pad, 0)
    high = np.minimum(np.array([x.max() for x in coords]) + pad + 1, ct.shape)
    box = tuple(slice(int(a), int(b)) for a, b in zip(low, high))
    heart_roi, coronary_roi, fat_roi, ct_roi = heart[box], coronary[box], torso_fat[box], ct[box]
    layer_union = np.zeros(heart_roi.shape, dtype=bool)
    for layer_path in sorted(Path(args.layers_dir).glob("L*.nii.gz")):
        layer_image, layer_array = read(layer_path)
        check_grid(ct_image, layer_image, str(layer_path))
        layer_union |= layer_array[box] > 0
    d_heart = ndi.distance_transform_edt(~heart_roi, sampling=spacing)
    d_coronary = ndi.distance_transform_edt(~coronary_roi, sampling=spacing)
    validated_hu = (ct_roi >= -190) & (ct_roi <= -30)
    shell = fat_roi & validated_hu & ~heart_roi & (d_heart <= 4)
    near_coronary = shell & (d_coronary <= 8)
    internal_gap = fat_roi & validated_hu & heart_roi & ~layer_union
    internal_coronary_gap = internal_gap & (d_coronary <= 8)

    # Present candidates as diagnostics, not a final L7 mask.
    all_masks = {}
    for name, local in (("heart_shell", shell), ("coronary_near", near_coronary),
                        ("internal_gap", internal_gap), ("internal_coronary_gap", internal_coronary_gap)):
        full = np.zeros(ct.shape, dtype=np.uint8)
        full[box] = local
        all_masks[name] = full > 0
        image = sitk.GetImageFromArray(full)
        image.CopyInformation(ct_image)
        sitk.WriteImage(image, str(output / f"{name}_diagnostic.nii.gz"), True)

    voxel_ml = float(np.prod(ct_image.GetSpacing()) / 1000)
    metrics = {
        "task": "TotalSegmentator tissue_types, torso_fat plus HU/heart constraints",
        "note": "diagnostic candidates, not approved L7 segmentation",
        "raw_torso_fat_ml": float(torso_fat.sum() * voxel_ml),
        "heart_shell_ml": float(shell.sum() * voxel_ml),
        "coronary_near_ml": float(near_coronary.sum() * voxel_ml),
        "internal_gap_ml": float(internal_gap.sum() * voxel_ml),
        "internal_coronary_gap_ml": float(internal_coronary_gap.sum() * voxel_ml),
        "shell_hu_percentiles": np.percentile(ct_roi[shell], [1, 5, 25, 50, 75, 95, 99]).tolist() if shell.any() else [],
        "near_hu_percentiles": np.percentile(ct_roi[near_coronary], [1, 5, 25, 50, 75, 95, 99]).tolist() if near_coronary.any() else [],
    }
    for name, mask in all_masks.items():
        counts = mask.sum(axis=(1, 2))
        bins = [np.arange(start, min(start + 30, len(counts))) for start in (55, 85, 115, 145, 175, 205)]
        selected = [int(b[np.argmax(counts[b])]) for b in bins]
        panels = [overlay(ct, heart, coronary, mask, z) for z in selected]
        w, h = panels[0].size
        montage = Image.new("RGB", (w * 3, h * 2), (15, 15, 15))
        for i, panel in enumerate(panels):
            montage.paste(panel, ((i % 3) * w, (i // 3) * h))
        montage.save(output / f"{name}_axial_review.png")
        metrics[f"{name}_review_slices"] = selected

    (output / "review_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
