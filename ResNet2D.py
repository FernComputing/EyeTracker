#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 21:48:31 2024

@author: garrett
"""

import torch
import torch.nn as nn


class ResCon2D(nn.Module):
    def __init__(self,  in_channels, out_channels, kernel_size, stride, **kwargs):
        super().__init__()
        
        self.x1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(in_channels))
        
        self.x2 = self.x1
        
        self.x3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(in_channels))
        
        self.x4 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(in_channels),
            nn.MaxPool2d((in_channels,in_channels)))
        
    def forward(self, x):
        x1 = self.x1(x) + x
        x2 = self.x2(x1) + x
        x3 = self.x3(x2) + x
        x4 = self.x4(x3)
        
        return x4

class latlayer(nn.Module):
    def __init__(self, cov):
        super().__init__()
        self.cov = cov
        
    def forward(self, x, cc):
        cc = torch.full(x.shape, cc).to('cuda')
        covvy = torch.diag(self.cov) * torch.eye(28).to('cuda')
        out = x + x * cc @ (self.cov - covvy)
        # print(covvy.shape)
        return out
    
class latbuild(nn.Module):
    def __init__(self, cov=None, in_channels=1, n_classes = 10):
        super().__init__()
        self.n_classes = n_classes
        self.cov = cov.to('cuda')
        self.lats_on = True
        
        self.x0 = ResCon2D(in_channels=in_channels, out_channels=1, kernel_size=(3,3), stride=0, padding =0)
        self.x1 = nn.Sequential(
            nn.Linear(81, 28),
            nn.ReLU())
        
        self.xlat = latlayer(self.cov)
        
        self.x2 = nn.Sequential(
            nn.Linear(28, n_classes),
            nn.Softmax(dim=0))

    
    def forward(self, x):
        x0 = self.x0(x)
        x1 = torch.flatten(x0, start_dim=1)
        x2 = self.x1(x1)
        if self.lats_on:
            x3 = self.xlat(x2, 0.1)
            out = self.x2(x3)
        else:
            out = self.x2(x2)
        
        return out, x2

class nolatbuild(nn.Module):
    def __init__(self,  in_channels=3, n_classes = 2):
        super().__init__()
        self.n_classes = n_classes
             
        self.l0 = ResCon2D(in_channels=in_channels, out_channels=1, kernel_size=(3,3), stride=1, padding =0)
        self.l1 = nn.Sequential(
            nn.Linear(49, 32),
            nn.ReLU())
        
        self.l2 = nn.Sequential(
            nn.Linear(32, n_classes))
            # nn.Softmax(dim=0))

    
    def forward(self, x):
        x0 = self.l0(x)
        x1 = torch.flatten(x0, start_dim=1)
        x2 = self.l1(x1)
        out = self.l2(x2)
        
        return out, x2
    

      