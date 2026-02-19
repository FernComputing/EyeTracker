#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 19:02:52 2025

@author: garrett
"""

import torch
from torch.nn import functional as F
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np

norm = Normal(0, 1)
device = torch.device("cuda")
class generator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, kernel_size=(3,3), padding=1):
        super(generator, self).__init__()
        self.cc =torch.tensor(0.01).to(device)
        self.lats_on=True
        self.x1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=(3,3), padding=1),
            nn.Linear(28, 28),
            # nn.LeakyReLU(),
            nn.GELU(),
            # nn.LogSoftmax()            
            nn.BatchNorm2d(1),
            )
        self.x2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=(3,3), padding=1),
            nn.Linear(28, 28),
            # nn.LeakyReLU(),
            nn.GELU(),
            # nn.Softmax()
            # nn.Sigmoid()
            # nn.AvgPool2d(kernel_size=(28,28))
            nn.BatchNorm2d(1),
            )
        self.x3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=(3,3), padding=1),
            nn.Linear(28, 28),
            # nn.LeakyReLU(),
            nn.GELU(),
            nn.BatchNorm2d(1),
            )
        
        # self.lats = norm.sample((28**2,28**2)).to(device)
        # self.lats = nn.Parameter(0.25 * gaussian_corr_matrix(28**2, sigma=28).to(device))
        self.lats =nn.Parameter(norm.sample((28**2, 28**2)))
        self.lats.requires_grad = True
        
        self.lats2 = nn.Parameter(0.25 * gaussian_corr_matrix(28**2, sigma=28).to(device))
        self.lats2.requires_grad = True
        # lats_dist = MultivariateNormal(loc = torch.zeros(28**2), covariance_matrix=corr_mat)
        # self.lats = lats_dist.sample(28**2)
        
    def forward(self, x):
        """
        x: (B, 1, 28, 28)
        returns: (x2, x0)
        """
        x0 = self.x1(x)
        if self.lats_on:
            B = x0.shape[0]

            xa = x0.reshape(B, 28**2)          # (B, N)
            xb = xa @ (0.5 * self.lats.T)              # (B, N)
            xc = xb.reshape_as(x0)             # (B, 1, 28, 28)
            xlat = x0 + self.cc * xc

            x1 = self.x2(xlat.clone().detach())
            x1a = x1.reshape(B, 28**2)
            x1b = x1a @ self.lats2.T
            x1c = x1b.reshape_as(x1)
            x1lat = x1 + self.cc * x1c

            x2 = self.x3(x1lat)
        else:
            x1 = self.x2(x0.clone().detach())
            x2 = self.x3(x1)

        return x2, x0
    

def gaussian_corr_matrix(n, sigma):
    idx = torch.arange(n)
    dist = torch.tensor(np.subtract.outer(idx, idx))
    return torch.exp(-0.5 * (dist/sigma)**2)
