#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 20:32:33 2025

@author: garrett
"""

import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F 
import os
import h5py

class MI_MNIST(Dataset):
    
    def __init__(self, path=None,
                train=None,
                test=None,
                batch = None,
                device=None):
        
        self.path = path
        self.idpath = os.listdir(path=self.path)
        self.idpath.remove('MI_all.h5')
        
    def __len__(self):
        return len(self.idpath)
    
    
    def __getitem__(self, idx):
       
       with h5py.File((self.path+'/'+self.idpath[idx]), 'r') as f:
           im = torch.tensor(f['image'][:])
           mi_map =torch.tensor(f['MI_map'][:])
       idp = self.idpath[idx]
       lab = F.one_hot(torch.tensor(int(idp[2])),num_classes=10)
       
       self.samp = {"image": im.float(), 'mi_map': mi_map.float(), 'label':lab.float()}
       return self.samp
           