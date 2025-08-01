#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 10:46:52 2025

@author: garrett
"""

import torch
import numpy as np

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
    # wnew = (alpha * error * )
    
    return wnew
     