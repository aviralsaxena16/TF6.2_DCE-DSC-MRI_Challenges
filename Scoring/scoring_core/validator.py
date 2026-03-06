import re
import nibabel as nib
import numpy as np
from pathlib import Path


# Naming convention expected by the current challengeScoring implementation
# Example: Clinical_P1_Visit1.nii or Synthetic_P2_Visit2.nii.gz
VALID_PATTERN = re.compile(
    r"(Clinical|Synthetic)_P\d+_Visit\d+\.nii(\.gz)?$",
    re.IGNORECASE,
)


def validate_submission(submission_dir: str) -> dict:
    """
    Performs lightweight validation of an OSIPI challenge submission.

    Checks include:
    - submission directory existence
    - presence of NIfTI files
    - filename conventions used by the scoring script
    - NIfTI readability
    - basic parameter map sanity checks
    - duplicate patient visit detection

    Returns
    -------
    dict
        {
            "valid": bool,
            "file_count": int,
            "errors": list
        }
    """

    errors = []
    nifti_files = []
    submission_path = Path(submission_dir)

    if not submission_path.exists():
        return {
            "valid": False,
            "file_count": 0,
            "errors": [f"Submission directory not found: {submission_dir}"],
        }

    for file_path in submission_path.rglob("*"):

        if file_path.suffix.lower() in [".nii", ".gz"] and str(file_path).lower().endswith((".nii", ".nii.gz")):

            filename = file_path.name
            nifti_files.append(filename)

            # Check filename pattern
            if not VALID_PATTERN.match(filename):
                errors.append(f"Unexpected filename format: {filename}")

            # Validate NIfTI readability
            try:
                img = nib.load(str(file_path))
                data = np.asarray(img.dataobj)

                # Basic parameter sanity checks
                if not np.isfinite(data).any():
                    errors.append(f"Invalid parameter values in {filename} (all NaN/Inf)")

            except Exception:
                errors.append(f"Invalid or corrupted NIfTI file: {filename}")

    if len(nifti_files) == 0:
        errors.append("No NIfTI files found in submission directory")

    seen = set()
    duplicates = []

    for f in nifti_files:
        base = f.replace(".nii.gz", "").replace(".nii", "")
        if base in seen:
            duplicates.append(base)
        seen.add(base)

    if duplicates:
        errors.append(f"Duplicate patient visit files detected: {duplicates}")

    return {
        "valid": len(errors) == 0,
        "file_count": len(nifti_files),
        "errors": errors,
    }