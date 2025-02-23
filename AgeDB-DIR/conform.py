import torch

#
# return the absolute residu of the prediction
#
# tau is a fixed number for estimating the upper and lower bound
def abs_err(model, loader, tau):
    device = next(model.parameters()).device
    with torch.no_grad():
        for idx, (x, _) in enumerate(loader):
            x = x.to(device)
            y_pred, lower, upper, _ = model(x)
            #lower, upper  =  torch.abs(lower) , torch.abs(upper)
            err = torch.max(lower - y_pred, y_pred - upper)
            abs_err, _ = torch.sort(err, dim=0)
            idx = int((1-tau)*abs_err.shape[0])
            q = abs_err[idx]
            interval = upper - lower  + 2*q
            #abs_err, abs_idx = torch.sort(abs, -1), torch.argsort(abs, -1)
    #print(f' the first {interval[:10]}')
    return interval


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