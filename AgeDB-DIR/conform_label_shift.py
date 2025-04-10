import torch

def lscp_quantile(train_targets, calib_preds, calib_targets, num_bins=10, alpha=0.1):
    """
    train_targets: (N_train,) — used to estimate π_S
    calib_preds, calib_targets: (N_calib,) — from model evaluated on calibration set
    """

    # 1. Bin targets (shared bins for train/calib)
    all_targets = torch.cat([train_targets, calib_targets])
    min_val, max_val = all_targets.min(), all_targets.max()
    bins = torch.linspace(min_val, max_val, num_bins + 1)

    def bin_targets(y):
        return torch.clamp(torch.bucketize(y, bins) - 1, 0, num_bins - 1)

    train_bins = bin_targets(train_targets)
    calib_bins = bin_targets(calib_targets)

    # 2. Estimate π_S and π_T
    pi_S = torch.bincount(train_bins, minlength=num_bins).float()
    pi_T = torch.bincount(calib_bins, minlength=num_bins).float()
    pi_S /= pi_S.sum()
    pi_T /= pi_T.sum()

    # 3. Compute residuals on calibration set
    residuals = torch.abs(calib_preds - calib_targets)

    # 4. Compute weights: w_i = pi_S / pi_T
    weights = pi_S[calib_bins] / (pi_T[calib_bins] + 1e-8)

    # 5. Compute weighted quantile (1 - alpha coverage)
    sorted_residuals, sorted_idx = torch.sort(residuals)
    sorted_weights = weights[sorted_idx]
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