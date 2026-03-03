import numpy as np


def compute_cov(visit1_value, visit2_value):
    """
    Compute Coefficient of Variation (CoV)
    """
    mean_val = (visit1_value + visit2_value) / 2
    if mean_val == 0:
        return np.nan
    return np.std([visit1_value, visit2_value]) / mean_val


def compute_rmse(predicted, ground_truth, mask):
    """
    Compute Root Mean Squared Error within mask
    """
    masked_diff = (predicted - ground_truth)[mask > 0]
    return np.sqrt(np.mean(masked_diff ** 2))


def compute_repeatability(visit1_value, visit2_value):
    """
    Repeatability calculation logic (as defined in original script)
    """
    mean_val = (visit1_value + visit2_value) / 2
    if mean_val == 0:
        return np.nan

    std_val = np.std([visit1_value, visit2_value])
    return std_val / mean_val


def compute_accuracy(predicted, ground_truth, mask):
    """
    Accuracy metric (fractional variation or RMSE depending on implementation)
    """
    masked_pred = predicted[mask > 0]
    masked_gt = ground_truth[mask > 0]

    if masked_gt.size == 0:
        return np.nan

    return np.mean(np.abs(masked_pred - masked_gt) / masked_gt)


def compute_reproducibility(score_list):
    """
    Reproducibility metric placeholder extracted from original script logic.
    """
    if len(score_list) == 0:
        return np.nan

    return np.mean(score_list)