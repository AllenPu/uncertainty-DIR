import torch
import torch.nn as nn
import torch.nn.functional as F



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
    pred = model(x)
    lower = pred - q_hat
    upper = pred + q_hat
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
        prediction : torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
        lamb: float
    ):
    #
    bound_penalty = F.relu(lower - prediction) + F.relu(prediction - upper)
    #coverage
    coverage_penalty = upper - lower
    loss = bound_penalty.mean() + coverage_penalty.mean()
    #
    return loss
    