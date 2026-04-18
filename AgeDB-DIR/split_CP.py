import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def calibrate_qhat_from_batch(model, cal_batch, device, alpha=0.1):
    was_training = model.training
    model.eval()
    with torch.no_grad():
        x_cal, y_cal, _ = cal_batch
        x_cal, y_cal = x_cal.to(device), y_cal.to(device)
        y_pred, lower, upper, _ = model(x_cal)
        score = torch.maximum(lower - y_cal, y_cal - upper)
        score = torch.clamp(score, min=0.0)
        q_hat = torch.quantile(score.flatten(), 1 - alpha)
    model.train(was_training)
    return q_hat


def predict_with_interval(
    model: nn.Module,
    x: torch.Tensor,
    q_hat: float,
    device: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns point prediction, lower bound, upper bound.
    """
    model.eval()
    x = x.to(device)
    pred, lower, upper, _ = model(x)
    lower = lower - q_hat
    upper = upper + q_hat
    return pred.cpu(), lower.cpu(), upper.cpu()



def evaluate_conformal(
    model: nn.Module,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    q_hat: float,
    device: str
) -> Tuple[float, float]:
    """
    Returns:
    - empirical coverage
    - average interval width
    """
    pred, lower, upper = predict_with_interval(model, x_test, q_hat, device)

    y_test_cpu = y_test.cpu()
    covered = ((y_test_cpu >= lower) & (y_test_cpu <= upper)).float()
    coverage = covered.mean().item()

    avg_width = (upper - lower).mean().item()
    return coverage, avg_width



def split_cp_loss(
        loss: nn.Module,
        true: torch.Tensor,
        prediction: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
        lamb: float
    ):
    #
    bound_penalty = F.relu(lower - prediction) + F.relu(prediction - upper)
    #coverage
    k=10
    soft_covered = torch.sigmoid(k * (true - lower)) * torch.sigmoid(k * (upper - true))
    coverage = soft_covered.mean()
    coverage_penalty = torch.relu(lamb - coverage)
    #
    loss = bound_penalty.mean() + coverage_penalty.mean()
    #
    return loss
