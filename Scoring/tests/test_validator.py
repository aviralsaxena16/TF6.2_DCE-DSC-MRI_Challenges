import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from scoring_core.validator import validate_submission


class TestSubmissionValidator:
    """
    Unit tests for submission validation logic.
    Uses pytest temporary directories and mocks
    to simulate NIfTI loading without heavy I/O.
    """

    def test_empty_directory(self, tmp_path):
        """Empty directory should fail validation."""
        result = validate_submission(str(tmp_path))

        assert result["valid"] is False
        assert result["file_count"] == 0

    @patch("nibabel.load")
    def test_valid_submission(self, mock_load, tmp_path):
        """Correct filenames should pass validation."""

        (tmp_path / "Clinical_P1_Visit1.nii").touch()
        (tmp_path / "Synthetic_P1_Visit1.nii").touch()

        mock_img = MagicMock()
        mock_img.get_fdata.return_value = np.ones((10, 10))
        mock_load.return_value = mock_img

        result = validate_submission(str(tmp_path))

        assert result["file_count"] == 2

    @patch("nibabel.load")
    def test_invalid_filename(self, mock_load, tmp_path):
        """Incorrect filename formats should trigger errors."""

        (tmp_path / "invalid_name.nii").touch()

        mock_img = MagicMock()
        mock_img.get_fdata.return_value = np.ones((10, 10))
        mock_load.return_value = mock_img

        result = validate_submission(str(tmp_path))

        assert result["valid"] is False
        assert len(result["errors"]) > 0

    @patch("nibabel.load")
    def test_nan_tensor_detection(self, mock_load, tmp_path):
        """Validator should detect arrays with only NaN values."""

        (tmp_path / "Clinical_P1_Visit1.nii").touch()

        mock_img = MagicMock()
        mock_img.get_fdata.return_value = np.full((5, 5), np.nan)
        mock_load.return_value = mock_img

        result = validate_submission(str(tmp_path))

        assert result["valid"] is False