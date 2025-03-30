import os
#import logging
import numpy as np
from PIL import Image
from scipy.ndimage import convolve1d
from torch.utils import data
import torchvision.transforms as transforms
import pandas as pd
from torch.utils.data import DataLoader
from utils import get_lds_kernel_window, shot_count, check_shot
import math
import torch
from PIL import ImageFilter 
import random


class AgeDB(data.Dataset):
    def __init__(self, df, data_dir, img_size, split='train', group_num=10, reweight='inverse', smooth = 'lds', max_age=100, aug=False):
        self.df = df
        self.data_dir = data_dir
        self.img_size = img_size
        self.split = split
        self.group_list = []
        self.group_num = group_num
        #
        self.aug = aug
        #
        self.split = split
        #
        self.y_min, self.y_max = np.max(df['age']), np.min(np.max(df['age']))
        self.smooth = smooth
        self.range_vals =torch.linspace(self.y_min, self.y_max, self.group_num)
        #
        #print(self.split)
        #
        if self.split == 'train' and self.smooth == 'lds':
            self.weights = self._prepare_weights(reweight, smooth = self.smooth)
        #
        print(f' reweight is {reweight} and smooth is {smooth}')
           

    def __len__(self):
        return len(self.df)
    



    # add to test the multi expert
    def __getitem__(self, index):
        index = index % len(self.df)
        row = self.df.iloc[index]
        img = Image.open(os.path.join(
            self.data_dir, row['path'])).convert('RGB')
        transform = self.get_transform()

        imgs = transform(img)

        label = np.asarray([row['age']]).astype('float32')

        if self.split == 'train' and self.reweight != 'none':
            weight = np.asarray([self.weights[index]]).astype('float32') if self.weights is not None else np.asarray([np.float32(1.)])
            return imgs, label, weight
        else:
            return imgs, label, 1
        
        
    # return a dictionary, key : label, value : corresponding  weights.
    def get_weight_dict(self):
        weight_label_dict = {}
        labels = self.df['age'].values
        for (w, l) in zip(self.weights, labels):
            weight_label_dict[l] = weight_label_dict.get(l, int(w))
        return weight_label_dict

    

    def get_transform(self):
        if self.split == 'train':
            transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomCrop(self.img_size, padding=16),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([.5, .5, .5], [.5, .5, .5]),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize([.5, .5, .5], [.5, .5, .5]),
            ])
        return transform


    # return additioanl transform
    def aug_transform(self):
        train_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.RandomCrop(self.img_size, padding=4),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.2),
            transforms.RandomHorizontalFlip(), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],std=[0.2023, 0.1994, 0.2010]),
            ])
        return train_transform
    
    

    def _prepare_weights(self, reweight, smooth='lds', max_target=121, lds_kernel='gaussian', lds_ks=5, lds_sigma=2):
        if smooth == 'lds':
            lds = True
        else:
            lds = False
        print(f' Enabling LDS smoothness is {lds}')

        assert reweight in {'none', 'inverse', 'sqrt_inv'}
        #
        #assert reweight != 'none' if lds else True, \
        #    "Set reweight to \'sqrt_inv\' (default) or \'inverse\' when using LDS"
        #

        value_dict = {x: 0 for x in range(max_target)}
        labels = self.df['age'].values
        for label in labels:
            value_dict[min(max_target - 1, int(label))] += 1
        if reweight == 'sqrt_inv':
            value_dict = {k: np.sqrt(v) for k, v in value_dict.items()}
        elif reweight == 'inverse':
            # clip weights for inverse re-weight
            value_dict = {k: np.clip(v, 5, 1000)
                          for k, v in value_dict.items()}
        num_per_label = [
            value_dict[min(max_target - 1, int(label))] for label in labels]
        if not len(num_per_label) :  #or reweight == 'none':
            return None
        #
        print(f"Using re-weighting: [{reweight.upper()}]")
        #
        # if lds and reweight is both none, return weitgh = 1
        #
        if lds:
            lds_kernel_window = get_lds_kernel_window(
                lds_kernel, lds_ks, lds_sigma)
            print(f'Using LDS: [{lds_kernel.upper()}] ({lds_ks}/{lds_sigma})')
            smoothed_value = convolve1d(
                np.asarray([v for _, v in value_dict.items()]), weights=lds_kernel_window, mode='constant')
            num_per_label = [
                smoothed_value[min(max_target - 1, int(label))] for label in labels]
        #
        if reweight != 'none' or lds:
            weights = [np.float32(1 / x) for x in num_per_label]
            scaling = len(weights) / np.sum(weights)
            weights = [scaling * x for x in weights]
        # if not lds set all 1 (equal weght)
        else:
            weights = [1 for x in num_per_label]
        #
        #print(f"--{self.split}---{reweight}-----{lds}-----")
        #
        return weights
    




class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

