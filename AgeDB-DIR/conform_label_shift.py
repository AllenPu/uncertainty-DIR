import torch

def lscp_quantile(weights, calib_preds, calib_targets, alpha=0.1):
    """
    weights (N_train,  1) — estimate the weight on each calibration label
    calib_preds, calib_targets: (N_calib,) — from model evaluated on calibration set
    """

    # 1. Compute residuals on calibration set
    residuals = torch.abs(calib_preds - calib_targets)
    cal_weights = torch.Tensor(weights[l.item()] for l in calib_targets)

    # 2. Compute weighted quantile (1 - alpha coverage)
    sorted_residuals, sorted_idx = torch.sort(residuals)
    sorted_weights = cal_weights[sorted_idx]
    cum_weights = torch.cumsum(sorted_weights, dim=0)
    cutoff = (1 - alpha) * cum_weights[-1]
    idx = torch.searchsorted(cum_weights, cutoff)
    q = sorted_residuals[min(idx, len(sorted_residuals)-1)]

    return q



def construct_prediction_intervals(test_preds, q):
    lower = test_preds - q
    upper = test_preds + q
    return torch.stack([lower, upper], dim=1)


def cal_interval(train_y, calib_preds, calib_y, test_preds):
    q = lscp_quantile(train_y, calib_preds, calib_y, num_bins=10, alpha=0.1)
    intervals = construct_prediction_intervals(test_preds, q)
    return intervals