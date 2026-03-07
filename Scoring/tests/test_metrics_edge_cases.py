import numpy as np
import pytest

from scoring_core.metrics import compute_cov


class TestMetricsEdgeCases:
    """
    Tests numerical stability and edge-case behaviour
    of scoring metrics.
    """

    def test_zero_arrays(self):
        """Metric should not crash when both arrays are zero."""
        arr1 = np.zeros((5, 5))
        arr2 = np.zeros((5, 5))

        try:
            compute_cov(arr1, arr2)
        except ZeroDivisionError:
            pytest.fail("compute_cov crashed on zero arrays")

    def test_nan_values(self):
        """Metric should handle NaN values gracefully."""
        arr1 = np.array([1.0, np.nan, 3.0])
        arr2 = np.array([1.0, 2.0, 3.0])

        try:
            compute_cov(arr1, arr2)
        except Exception as e:
            pytest.fail(f"compute_cov failed with NaN values: {e}")

    def test_inf_values(self):
        """Metric should not crash on infinite values."""
        arr1 = np.array([1.0, np.inf, 3.0])
        arr2 = np.array([1.0, 2.0, 3.0])

        try:
            compute_cov(arr1, arr2)
        except Exception as e:
            pytest.fail(f"compute_cov failed with Inf values: {e}")