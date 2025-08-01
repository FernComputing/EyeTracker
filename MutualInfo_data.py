#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 19 19:28:09 2025

@author: garrett
"""

import torch
from sklearn.metrics import mutual_info_score

def compute_pixelwise_mi(images, labels, n_bins=16):
    """
    images: numpy array of shape [N, H, W] or [N, H, W, C] (grayscale or RGB)
    labels: numpy array of shape [N]
    returns: MI map of shape [H, W]
    """
    N, C, H, W = images.shape[:]
  
    images = images.mean(axis=1)
    
    # Flatten to [N, H*W]
    flat_images = images.reshape(N, -1)
    
    # Discretize pixel values to bins
    binned = torch.floor((flat_images - flat_images.min()) / (flat_images.max() - flat_images.min() + 1e-8) * (n_bins - 1))
    
    mi_map = torch.zeros((flat_images.shape[1],))
    
    for i in range(flat_images.shape[1]):
        mi = mutual_info_score(binned[:, i], labels)
        mi_map[i] = mi
    
    # Reshape to [H, W]
    return mi_map.reshape(H, W)

    