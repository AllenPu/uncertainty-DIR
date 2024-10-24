import torch
import numpy as np
import torch.nn as nn



def beta_nll_loss(mean, variance, target, beta=0.5):
    """Compute beta-NLL loss
    
    :param mean: Predicted mean of shape B x D
    :param variance: Predicted variance of shape B x D
    :param target: Target of shape B x D
    :param beta: Parameter from range [0, 1] controlling relative 
        weighting between data points, where `0` corresponds to 
        high weight on low error points and `1` to an equal weighting.
    :returns: Loss per batch element of shape B
    """
    loss = 0.5 * ((target - mean) ** 2 / variance + variance.log())

    if beta > 0:
        loss = loss * (variance.detach() ** beta)
    loss = torch.sum(loss)
    return loss



# estimate the p(z_d)
class KNIFE(nn.Module):
    def __init__(self, args, zc_dim, zd_dim):
        super(KNIFE, self).__init__()
        self.kernel_marg = MargKernel(args, zc_dim, zd_dim)

    def forward(self, z_c, z_d):  # samples have shape [sample_size, dim]
        marg_ent = self.kernel_marg(z_d)
        return marg_ent

    def learning_loss(self, z_c, z_d):
        marg_ent = self.kernel_marg(z_d)
        return marg_ent 




class MargKernel(nn.Module):
    """
    Used to compute p(z_d)
    """

    def __init__(self, args, zc_dim, init_samples=None):

        self.optimize_mu = args.optimize_mu
        self.K = args.marg_modes if self.optimize_mu else args.batch_size
        self.d = zc_dim
        # K is the batch size, d is the feature dimentsion
        self.init_std = 0.01
        super(MargKernel, self).__init__()
        self.logC = torch.tensor([-self.d / 2 * np.log(2 * np.pi)])
        # mean : a
        init_samples = self.init_std * torch.randn(self.K, self.d)
        self.means = nn.Parameter(init_samples, requires_grad=True)  # [K, db]
        # var : A
        diag = self.init_std * torch.randn((1, self.K, self.d))
        self.logvar = nn.Parameter(diag, requires_grad=True)
        #
        tri = self.init_std * torch.randn((1, self.K, self.d, self.d))
        tri = tri.to(init_samples.dtype)
        self.tri = nn.Parameter(tri, requires_grad=True)
        # weight : w
        weigh = torch.ones((1, self.K))
        self.weigh = nn.Parameter(weigh, requires_grad=True)


    def logpdf(self, x):
        assert len(x.shape) == 2 and x.shape[1] == self.d, 'x has to have shape [N, d]'
        x = x[:, None, :]
        w = torch.log_softmax(self.weigh, dim=1)
        #
        y = x - self.means
        #
        logvar = self.logvar
        #
        var = logvar.exp()
        y = y * var
        # print(f"Marg : {var.min()} | {var.max()} | {var.mean()}")
        y = y + torch.squeeze(torch.matmul(torch.tril(self.tri, diagonal=-1), y[:, :, :, None]), 3)
        #
        y = torch.sum(y ** 2, dim=2)
        #
        y = -y / 2 + torch.sum(torch.log(torch.abs(var) + 1e-8), dim=-1) + w
        y = torch.logsumexp(y, dim=-1)
        return self.logC.to(y.device) + y

    def update_parameters(self, z):
        self.means = z

    def forward(self, x):
        y = -self.logpdf(x)
        return torch.mean(y)