import torch

#
# return the absolute residu of the prediction
#
# tau is a fixed number for estimating the upper and lower bound
# train_weight_dict : a dictionary, key :label, value : weight in training
#
def abs_err(model, cal_batch, train_weight_dict, tau):
    device = next(model.parameters()).device
    with torch.no_grad():
        #print(cal_batch)
        x, y, _ = cal_batch
        #for idx, (x, _, w) in enumerate(loader):
        x = x.to(device)
        y_pred, lower, upper, _ = model(x)
        #lower, upper  =  torch.abs(lower) , torch.abs(upper)
        err = torch.max(lower - y_pred, y_pred - upper)
        nans =  torch.where(torch.isnan(err) == True)[0].tolist()
        if len(nans) != 0:
            err_ = err.squeeze(-1)
            for e in nans:
                print(f' y is {y[e]} y pred is {y_pred[e]} upper {upper[e]} lower {lower[e]} err is {err_[e]}')
            assert 1 == 2
        #
        element = [train_weight_dict[x.item()] for x in y]
        w = torch.tensor(element, dtype=torch.long).unsqueeze(-1)
        #print(f" the shape of w is {w.shape}, the shape of err is {err.shape}")
        err *= w.to(device)
        #
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