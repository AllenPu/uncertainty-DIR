import torch

#
# return the absolute residu of the prediction
#
def abs_err(model, loader):
    with torch.no_grad():
        for idx, (x, y) in enumerate(loader):
            y_pred, _, _ = model(x)
            abs = torch.abs(y_pred - y)
            abs_err, abs_idx = torch.sort(abs, -1), torch.argsort(abs, -1)
    return abs_err, abs_idx




def pinball_loss(y_true, y_pred, tau=0.5):
    """
    Compute the pinball loss (quantile loss) for given quantile tau.
    
    Parameters:
        y_true (Tensor): Ground truth values.
        y_pred (Tensor): Predicted values.
        tau (float): Quantile level (0 < tau < 1).
    
    Returns:
        Tensor: Pinball loss.
    """
    loss = torch.where(y_true >= y_pred, tau * (y_true - y_pred), (1 - tau) * (y_pred - y_true))
    return loss.mean()