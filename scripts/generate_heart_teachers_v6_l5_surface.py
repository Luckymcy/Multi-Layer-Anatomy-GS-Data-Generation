import argparse
import json
import os
import shutil
import sys

import numpy as np
import SimpleITK as sitk


sys.path.append(os.path.dirname(__file__))
import generate_heart_teachers_v2 as legacy
import generate_heart_teachers_v5_sdf as v5


L4_COLOR = (0.56, 0.105, 0.085)
L5_COLOR = (0.68, 0.145, 0.115)
L6_COLOR = (0.30, 0.025, 0.030)
L4_SIGMA_MM = 1.00
L5_SDF_SIGMA_MM = 1.40
L5_HIDDEN_OVERLAP_MM = 0.30
L5_MESH_ITERATIONS = 30
L5_MESH_PASS_BAND = 0.05
L6_SIGMA_MM = 0.16


def mask_sdf(mask):
    sdf = sitk.SignedMaurerDistanceMap(
        sitk.Cast(mask > 0, sitk.sitkUInt8),
        insideIsPositive=True,
        squaredDistance=False,
        useImageSpacing=True,
    )
    # Restore the half-voxel boundary position that binary marching cubes uses.
    return sdf + min(mask.GetSpacing()) * 0.5


def build_l5_surfaces(l4_mask, l5_mask):
    l4_sdf = sitk.SmoothingRecursiveGaussian(mask_sdf(l4_mask), L4_SIGMA_MM)
    l5_training_sdf = sitk.SmoothingRecursiveGaussian(
        mask_sdf(l5_mask), L5_SDF_SIGMA_MM
    )

    # Positive-inside SDF intersection. L5 remains only where it is outside
    # L4 or at most 0.30 mm inside it, so the junction is hidden but the two
    # visible surfaces are never nearly coplanar over a broad region.
    l5_visible_sdf = sitk.Minimum(
        l5_training_sdf,
        L5_HIDDEN_OVERLAP_MM - l4_sdf,
    )
    training_surface = v5.surface_from_sdf(
        l5_training_sdf,
        L5_MESH_ITERATIONS,
        L5_MESH_PASS_BAND,
    )
    visible_surface = v5.surface_from_sdf(
        l5_visible_sdf,
        L5_MESH_ITERATIONS,
        L5_MESH_PASS_BAND,
    )
    return l5_training_sdf, l5_visible_sdf, training_surface, visible_surface


def copy_layers(source_root, output_root, last_layer):
    for layer_index in range(last_layer + 1):
        shutil.copytree(
            os.path.join(source_root, f"L{layer_index}"),
            os.path.join(output_root, f"L{layer_index}"),
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate L5/L6 teachers with a clipped, scale-preserving L5 "
            "surface while reusing L0-L4."
        )
    )
    parser.add_argument("--existing-teachers", required=True)
    parser.add_argument("--l4-outer-mask", required=True)
    parser.add_argument("--l5-training-mask", required=True)
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
    geometry_dir = os.path.join(args.output_dir, "geometry")
    os.makedirs(geometry_dir)

    with open(
        os.path.join(args.existing_teachers, "manifest.json"),
        "r",
        encoding="utf-8",
    ) as file:
        previous_manifest = json.load(file)

    copy_layers(args.existing_teachers, args.output_dir, 4)

    l4_mask = sitk.ReadImage(args.l4_outer_mask)
    l5_mask = sitk.ReadImage(args.l5_training_mask)
    l6_sdf = sitk.ReadImage(args.l6_sdf)
    if not (
        l4_mask.GetSize() == l5_mask.GetSize() == l6_sdf.GetSize()
        and l4_mask.GetSpacing() == l5_mask.GetSpacing() == l6_sdf.GetSpacing()
        and l4_mask.GetOrigin() == l5_mask.GetOrigin() == l6_sdf.GetOrigin()
        and l4_mask.GetDirection() == l5_mask.GetDirection() == l6_sdf.GetDirection()
    ):
        raise ValueError("L4, L5, and L6 geometry must match")

    l4_surface = v5.surface_from_mask(l4_mask, L4_SIGMA_MM, 12, 0.10)
    (
        l5_training_sdf,
        l5_visible_sdf,
        l5_training_surface,
        l5_visible_surface,
    ) = build_l5_surfaces(l4_mask, l5_mask)
    l6_surface = v5.surface_from_sdf(
        sitk.SmoothingRecursiveGaussian(l6_sdf, L6_SIGMA_MM),
        4,
        0.22,
    )

    l5_training_mask = sitk.Cast(l5_training_sdf > 0, sitk.sitkUInt8)
    l5_visible_mask = sitk.Cast(l5_visible_sdf > 0, sitk.sitkUInt8)
    training_mask_path = os.path.join(
        geometry_dir, "L5_training_overlap_iso.nii.gz"
    )
    visible_mask_path = os.path.join(
        geometry_dir, "L5_teacher_visible_iso.nii.gz"
    )
    sitk.WriteImage(l5_training_mask, training_mask_path, True)
    sitk.WriteImage(l5_visible_mask, visible_mask_path, True)

    l4_actor = v5.build_actor(l4_surface, L4_COLOR, 4)
    l5_actor = v5.build_actor(l5_visible_surface, L5_COLOR, 5)
    l6_actor = v5.build_actor(l6_surface, L6_COLOR, 6)

    camera = previous_manifest["camera"]
    center = np.asarray(camera["center"], dtype=float)
    radius = float(camera["radius"])
    view_angle = float(camera["view_angle_degrees"])
    base_count = int(camera["base_views"])
    extra_count = int(camera["extra_coronary_views"])
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

    l5_output = os.path.join(args.output_dir, "L5")
    l5_points = v5.render_layer(
        l5_output,
        [l4_actor, l5_actor],
        list(base_poses),
        l5_visible_surface,
        L5_COLOR,
        center,
        radius,
        args.output_size,
        args.render_size,
        view_angle,
        args.point_count,
        5,
        base_count,
    )
    l6_output = os.path.join(args.output_dir, "L6")
    l6_poses = list(base_poses) + list(extra_poses)
    l6_points = v5.render_layer(
        l6_output,
        [l4_actor, l5_actor, l6_actor],
        l6_poses,
        l6_surface,
        L6_COLOR,
        center,
        radius,
        args.output_size,
        args.render_size,
        view_angle,
        args.point_count,
        6,
        base_count,
    )

    layers = previous_manifest["layers"][:5]
    layers.extend(
        [
            {
                "index": 5,
                "name": "great_vessels",
                "training_mask": training_mask_path,
                "teacher_visible_mask": visible_mask_path,
                "surface_method": (
                    f"0.31 mm isotropic SDF; {L5_SDF_SIGMA_MM:.2f} mm "
                    "field smoothing; "
                    "30-pass windowed-sinc mesh smoothing"
                ),
                "hidden_overlap_inside_L4_mm": L5_HIDDEN_OVERLAP_MM,
                "color_rgb_0_1": list(L5_COLOR),
                "color_rgb_8bit": np.round(
                    np.asarray(L5_COLOR) * 255
                ).astype(int).tolist(),
                "views": base_count,
                "initial_points": l5_points,
                "teacher_visible_layers": [4, 5],
            },
            {
                "index": 6,
                "name": "coronary_arteries",
                "source": args.l6_sdf,
                "surface_method": "0.31 mm isotropic signed-distance field",
                "color_rgb_0_1": list(L6_COLOR),
                "color_rgb_8bit": np.round(
                    np.asarray(L6_COLOR) * 255
                ).astype(int).tolist(),
                "views": len(l6_poses),
                "initial_points": l6_points,
                "teacher_visible_layers": [4, 5, 6],
            },
        ]
    )
    manifest = {
        "generator": "generate_heart_teachers_v6_l5_surface.py",
        "based_on": os.path.join(args.existing_teachers, "manifest.json"),
        "camera": camera,
        "split": previous_manifest["split"],
        "rendering": {
            "render_size": args.render_size,
            "output_size": args.output_size,
            "downsample": "Lanczos",
            "material": "PBR",
            "L4_sdf_sigma_mm": L4_SIGMA_MM,
            "L5_sdf_sigma_mm": L5_SDF_SIGMA_MM,
            "L5_hidden_overlap_inside_L4_mm": L5_HIDDEN_OVERLAP_MM,
            "L5_mesh_iterations": L5_MESH_ITERATIONS,
            "L5_mesh_pass_band": L5_MESH_PASS_BAND,
            "L6_sdf_sigma_mm": L6_SIGMA_MM,
        },
        "layers": layers,
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
