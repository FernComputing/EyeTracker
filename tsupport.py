#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 12 20:43:17 2025

@author: garrett
"""

import torch
from torchvision.transforms import functional as F
import random as randy
from torchvision.datasets import OxfordIIITPet

device = torch.device("cuda")

def random_crop_with_coords(img, crop_size):
    h, w = img.size[1], img.size[0]
    th, tw = crop_size
    i = randy.randint(0, h - th)
    j = randy.randint(0, w - tw)
    cropped = F.crop(img, i, j, th, tw)
    return cropped, (i, j)

import torch

def crop_image(
    im: torch.Tensor,
    im_x=None,
    im_y=None,
    crop_size: int = 7,
    random: bool = False,
    device=None,
):
    """
    Batch-aware crop.
    im: (B, C, H, W) or (C, H, W)
    im_x, im_y: either ints or tensors of shape (B,) giving top-left x/y (w/h) coords
    returns: (cropt, x, y) if random=True else cropt
    """
    if device is None:
        device = im.device

    if im.dim() == 3:
        im = im.unsqueeze(0)  # (1, C, H, W)

    B, C, H, W = im.shape
    cs = int(crop_size)

    max_x = W - cs
    max_y = H - cs
    if max_x < 0 or max_y < 0:
        raise ValueError(f"crop_size={cs} larger than image size HxW={H}x{W}")

    if random:
        x = torch.randint(0, max_x + 1, (B,), device=device)
        y = torch.randint(0, max_y + 1, (B,), device=device)
    else:
        # accept python ints or tensors
        x = torch.as_tensor(im_x, device=device)
        y = torch.as_tensor(im_y, device=device)
        if x.dim() == 0: x = x.expand(B)
        if y.dim() == 0: y = y.expand(B)

        x = x.clamp(0, max_x)
        y = y.clamp(0, max_y)

    # Extract all cs x cs patches, then select per-sample index
    # patches: (B, C, H-cs+1, W-cs+1, cs, cs)
    patches = im.unfold(2, cs, 1).unfold(3, cs, 1)

    b_idx = torch.arange(B, device=device)
    cropt = patches[b_idx, :, y, x, :, :]  # (B, C, cs, cs)

    if random:
        return cropt, x, y
    return cropt

def add_patch_batch(int_im, x0, h, w, crop_size, mask=None):
    """
    int_im: (B, H, W)
    x0:     (B, 1, crop_size, crop_size) or (B, crop_size, crop_size)
    h, w:   (B,) top-left coords, long
    mask:   (B,) bool, optional
    """
    B, H, W = int_im.shape
    device = int_im.device

    if x0.dim() == 4:
        patch = x0[:, 0]   # -> (B, crop_size, crop_size)
    else:
        patch = x0

    if mask is not None:
        active = mask.nonzero(as_tuple=False).squeeze(1)
        if active.numel() == 0:
            return int_im
        h = h[active]
        w = w[active]
        patch = patch[active]
        batch_idx = active
    else:
        batch_idx = torch.arange(B, device=device)

    dh = torch.arange(crop_size, device=device)
    dw = torch.arange(crop_size, device=device)

    rows = h[:, None] + dh[None, :]          # (Ba, crop)
    cols = w[:, None] + dw[None, :]          # (Ba, crop)

    batch_idx = batch_idx[:, None, None]     # (Ba, 1, 1)
    rows = rows[:, :, None]                  # (Ba, crop, 1)
    cols = cols[:, None, :]                  # (Ba, 1, crop)

    int_im[batch_idx, rows, cols] += 3*patch
    return int_im

def crop_batch(im: torch.Tensor, w: torch.Tensor, h: torch.Tensor, crop_size: int):
    """
    Backward-compatible wrapper:
    w/h are (B,) tensors for the top-left coords.
    Returns (B, C, crop_size, crop_size).
    """
    return crop_image(im=im, im_x=w, im_y=h, crop_size=crop_size, random=False)
            
    

class CatVsDogDataset(OxfordIIITPet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cat_breeds = {
            'Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British_Shorthair',
            'Egyptian_Mau', 'Maine_Coon', 'Persian', 'Ragdoll', 'Russian_Blue',
            'Siamese', 'Sphynx'
        }

    def __getitem__(self, index):
        image, class_index = super().__getitem__(index)
        breed_name = self.classes[class_index - 1]  # class_index is 1-based
        label = 0 if breed_name in self.cat_breeds else 1  # 0: cat, 1: dog
        return image, label