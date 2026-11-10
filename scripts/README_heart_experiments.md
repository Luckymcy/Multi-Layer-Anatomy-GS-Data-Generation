# Heart CT experiment scripts

These optional scripts are separate from the upstream full-body/leg generation workflow. Patient datasets and derived geometry are not distributed with the repository.

- `build_l7_fat_preserved.py` regularizes an existing heart-adjacent torso-fat ROI with bounded support and volume preservation. It is not a new tissue detector. Its source ROI and CT must be supplied locally in the workspace layout used by the script.
- `build_l7_anatomical_candidates.py` creates CT-supported heart-adjacent-fat candidates for review, with an optional pericardium mask. Without that boundary, epicardial versus paracardial classification remains unvalidated.
- `smooth_l7_teacher_surface.py` smooths a supplied triangle mesh with a physical displacement bound and checks topology/volume. It does not modify the input mesh or validate tissue identity.
- `generate_l7_preserved_teachers.py` composites a supplied fat mesh with frozen L6 renders using approximate depth-based occlusion. Its optional `--lower-fill-intensity` defaults to zero to preserve existing lighting. It refuses an existing output directory and rejects an empty input mesh.

Workspace input paths now live under `data/ct/nifti`, `data/ct/masks` and `data/ct/teachers/v4`; `--root` remains the project workspace root. Some scripts retain development-dataset filenames/defaults, so inspect their input paths before using a different dataset. Models and generated outputs remain under separately supplied output locations.

Image-fit scores, fat-range HU support and mesh topology checks are distinct from clinical segmentation validation. These scripts do not implement real-image adversarial appearance training.
