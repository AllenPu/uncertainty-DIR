import torch.nn as nn
import torchvision
import torch
import torch.optim as optim
from model import *


class ResNet_regression(nn.Module):
    def __init__(self, args=None):
        super(ResNet_regression, self).__init__()
        self.args = args
        self.groups = args.groups
        exec('self.model = torchvision.models.resnet{}(pretrained=False)'.format(args.model_depth))
        #
        output_dim = args.groups * 2
        #
        fc_inputs = self.model.fc.in_features
        #
        self.model_extractor = nn.Sequential(*list(self.model.children())[:-1])
        #
        self.Flatten = nn.Flatten(start_dim=1)
        #
        self.model_linear =  nn.Sequential(nn.Linear(fc_inputs, output_dim))
        #

        #self.mode = args.mode
        self.sigma = args.sigma
        
    # g is the same shape of y
    def forward(self, x):
        #"output of model dim is 2G"
        z = self.model_extractor(x)
        #
        z = self.Flatten(z)
        #
        y_hat = self.model_linear(z)
        #
        # the ouput dim of the embed is : 512
        #
        return y_hat, z
    

# reparameterization for mu and sigma in ResNet
class reparam_ResNet(nn.Module):
    def __init__(self, name, args=None):
        super(ResNet_regression, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun()
        # mean ,std estimation networks
        self.emb2mu = nn.Linear(dim_in, 512)
        self.emb2std = nn.Linear(dim_in, 512)
        #


    def estimate(self, emb):
        """Estimates mu and std from the given input embeddings."""
        mean = self.emb2mu(emb)
        std = torch.nn.functional.softplus(self.emb2std(emb))
        return mean, std


    def reparameterize(self, mu, std):
        batch_size = mu.shape[0]
        z = torch.randn(self.sample_size, batch_size, mu.shape[1])
        return mu + std * z
    

    def forward(self, x, y):
        feat = self.encoder(x)
        mean, std = self.estimate(feat)
        z_reparameterized = self.reparameterize(mean, std)
        y_pred = self.regressor(feat)
        #
        return z_reparameterized, y_pred



#########################################################################################################

class Guassian_uncertain_ResNet(nn.Module):
    def __init__(self, name='resnet50', norm=False, weight_norm= False):
        super(Guassian_uncertain_ResNet, self).__init__()
        backbone, dim_in = model_dict[name]
        self.encoder = backbone()
        self.norm = norm
        self.weight_norm = weight_norm
        if self.weight_norm:
            self.regressor = torch.nn.utils.weight_norm(nn.Linear(dim_in, 2), name='weight')
        else:
           self.regressor = nn.Linear(dim_in, 2)
        self.guassian_head = GaussianLikelihoodHead(inp_dim=dim_in, outp_dim=1)
        self.feature_dim = dim_in        



    def forward(self, x):
        feat = self.encoder(x)
        if self.norm:
            feat = F.normalize(feat, dim=-1)
        out = self.guassian_head(feat)
        out_ = torch.chunk(out,2,dim=1)
        # feature, mean, variance
        return feat, out_[0], out_[1]
    




class GaussianLikelihoodHead(nn.Module):
    def __init__(
        self,
        inp_dim,
        outp_dim,
        initial_var=1,
        min_var=1e-8,
        max_var=100,
        mean_scale=1,
        var_scale=1,
        use_spectral_norm_mean=False,
        use_spectral_norm_var=False,
    ):
        super().__init__()
        assert min_var <= initial_var <= max_var

        self.min_var = min_var
        self.max_var = max_var
        self.init_var_offset = np.log(np.exp(initial_var - min_var) - 1)

        self.mean_scale = mean_scale
        self.var_scale = var_scale

        if use_spectral_norm_mean:
            self.mean = nn.utils.spectral_norm(nn.Linear(inp_dim, outp_dim))
        else:
            self.mean = nn.Linear(inp_dim, outp_dim)

        if use_spectral_norm_var:
            self.var = nn.utils.spectral_norm(nn.Linear(inp_dim, outp_dim))
        else:
            self.var = nn.Linear(inp_dim, outp_dim)

    def forward(self, inp):
        mean = self.mean(inp) * self.mean_scale
        var = self.var(inp) * self.var_scale

        var = F.softplus(var + self.init_var_offset) + self.min_var
        var = torch.clamp(var, self.min_var, self.max_var)

        return mean, var