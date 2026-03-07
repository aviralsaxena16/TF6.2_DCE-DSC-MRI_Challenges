import numpy as np
import pytest

from scoring_core.metrics import compute_cov


class TestMetricsCore:
    """
    Core correctness tests for OSIPI scoring metrics.
    Ensures expected behaviour for normal inputs.
    """

    def test_cov_identical_arrays(self):
        """CoV should be zero when arrays are identical."""
        arr1 = np.ones((10, 10)) * 5.0
        arr2 = np.ones((10, 10)) * 5.0

        cov = compute_cov(arr1, arr2)

        assert np.isclose(cov, 0.0), f"Expected 0.0 CoV, got {cov}"

    def test_cov_detects_variation(self):
        """CoV should be positive when arrays differ."""
        arr1 = np.full((5, 5), 10.0)
        arr2 = np.full((5, 5), 12.0)

        cov = compute_cov(arr1, arr2)

        assert cov > 0
        assert not np.isnan(cov)

    def test_cov_symmetry(self):
        """Metric should be symmetric: cov(a,b) == cov(b,a)."""
        a = np.random.rand(10, 10)
        b = np.random.rand(10, 10)

        cov1 = compute_cov(a, b)
        cov2 = compute_cov(b, a)

        assert np.isclose(cov1, cov2)