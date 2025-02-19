import torch

#
# return the absolute residu of the prediction
#
def abs_err(model, loader, tau):
    with torch.no_grad():
        for idx, (x, y) in enumerate(loader):
            y_pred, lower, upper = model(x)
            #lower, upper  =  torch.abs(lower) , torch.abs(upper)
            err = torch.max(lower - y_pred, y_pred - upper)
            abs_err, _ = torch.sort(err, dim=0)
            idx = int((1-tau)*abs_err.shape[0])
            q = abs_err[idx]
            q_len = upper - lower  + 2*q
            #abs_err, abs_idx = torch.sort(abs, -1), torch.argsort(abs, -1)
    return q_len


# low is the lower tau prediction
# high


# tau here is controlling the  
# tau_low = 0.5\alpha
# tau_high = 1 - 0.5\alpha
def pinball_loss(y_true, y_pred, tau=0.1):
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