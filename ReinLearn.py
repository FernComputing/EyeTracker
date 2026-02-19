#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 10:46:52 2025

@author: garrett
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def reinforcement_learning(alpha, error, inputs, acts):
    in_c = torch.mean(inputs, dim=1)
    in_m = torch.mean(inputs)
    # sc = torch.std(inputs, dim=2)
    acts_m = torch.mean(acts)
    Idelt = (in_c - in_m) 
    Idelt = Idelt.unsqueeze(0)
    actdelt = acts - acts_m
    actdelt = actdelt
    
    wnew = (alpha * error * Idelt @ actdelt)
    wnew /= torch.sqrt(abs(wnew))
    # wnew = (alpha * error * )
    
    return wnew
     
class search_learning(nn.Module):
    def __init__(self, lr=1e-3, kern_size=(3, 3), ndir=8):
        super().__init__()
        self.lr = lr
        self.kern_size = kern_size
        self.ndir = ndir

    def forward(self, sequence_info, mi_maps):
        """
        sequence_info : (B, T, 3)  -> [dir, w, h]
        mi_maps       : (B, H, W)
        """
        B, T, _ = sequence_info.shape
        k = self.kern_size[0]

        device = mi_maps.device
        total_loss = 0.0

        # loop over batch (safe, clear, correct)
        for b in range(B):

            seq = sequence_info[b]        # (T, 3)
            mi  = mi_maps[b]              # (H, W)

            tot_mi = mi.sum()
            loss_vec = torch.zeros(self.ndir, device=device)

            dirs = torch.unique(seq[:, 0]).long()

            for d in dirs:
                idx = torch.where(seq[:, 0] == d)[0]

                if len(idx) == 0:
                    continue

                maper = torch.zeros((k, k), device=device)

                for j in idx:
                    w = seq[j, 1].long()
                    h = seq[j, 2].long()

                    patch = mi[h:h+k, w:w+k]
                    maper += patch

                maper /= len(idx)

                # same objective you had
                loss_vec[d] = (maper.sum() - tot_mi) ** 2

            total_loss += loss_vec.sum()

        # scalar loss, scaled exactly like before
        return -self.lr * total_loss / B

        
    
    