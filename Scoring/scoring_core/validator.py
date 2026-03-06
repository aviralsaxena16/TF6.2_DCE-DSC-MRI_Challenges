import os
import re
import nibabel as nib
import numpy as np


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

    if not os.path.exists(submission_dir):
        return {
            "valid": False,
            "file_count": 0,
            "errors": [f"Submission directory not found: {submission_dir}"],
        }

    for root, _, files in os.walk(submission_dir):
        for file in files:

            if file.endswith(".nii") or file.endswith(".nii.gz"):

                filepath = os.path.join(root, file)
                nifti_files.append(file)

                # Check filename pattern
                if not VALID_PATTERN.match(file):
                    errors.append(f"Unexpected filename format: {file}")

                # Validate NIfTI readability
                try:
                    img = nib.load(filepath)
                    data = img.get_fdata()

                    # Basic parameter sanity checks
                    if not np.isfinite(data).any():
                        errors.append(f"Invalid parameter values in {file} (all NaN/Inf)")

                except Exception:
                    errors.append(f"Invalid or corrupted NIfTI file: {file}")

    if len(nifti_files) == 0:
        errors.append("No NIfTI files found in submission directory")

    return {
        "valid": len(errors) == 0,
        "file_count": len(nifti_files),
        "errors": errors,
    }