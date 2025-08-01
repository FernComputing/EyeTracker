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

def crop_image(im=None, im_x = None, im_y = None, crop_size = None, random = False):
    
    if len(im.shape)>0:
        im_w = im.shape[2]
        im_h = im.shape[3]
    
    if random: 
        
        iw = im_w - crop_size 
        ih = im_h - crop_size
        
        x = randy.randint(0, iw)
        x = torch.tensor(x).to(device)
        y = randy.randint(0, ih)
        y = torch.tensor(y).to(device)
        
        cropt = im[:, :, x:x+crop_size, y:y+crop_size]
        return cropt, x, y
    else:
        
        cropt = F.crop(im, im_y, im_x, crop_size, crop_size)
        
    if cropt.shape[2] != crop_size or cropt.shape[3] != crop_size:
        cropt_new = torch.zeros((cropt.shape[0], cropt.shape[1], crop_size, crop_size))
        cropt_new[:, :, :cropt.shape[2], :cropt.shape[3]] = cropt
        cropt = cropt_new

    
    return cropt
    
    
            
    

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