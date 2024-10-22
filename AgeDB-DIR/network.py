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
    


class uncertain_ResNet(nn.Module):
    def __init__(self, name, args=None):
        super(ResNet_regression, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun()
        # mean ,std estimation networks
        self.emb2mu = nn.Linear(dim_in, 512)
        self.emb2std = nn.Linear(dim_in, 512)
        # linear mapping to the regression
        self.regressor = nn.Linear(dim_in, 1)
    


    def estimate(self, emb):
        """Estimates mu and std from the given input embeddings."""
        mean = self.emb2mu(emb)
        std = torch.nn.functional.softplus(self.emb2std(emb))
        return mean, std


    def reparameterize(self, mu, std):
        batch_size = mu.shape[0]
        z = torch.randn(self.sample_size, batch_size, mu.shape[1])
        return mu + std * z
    

    def forward(self, x):
        feat = self.encoder(x)
        mean, std = self.estimate(feat)
        z_reparameterized = self.reparameterize(mean, std)
        y_pred = self.regressor(z_reparameterized)
        return z_reparameterized, y_pred


