#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 12 19:57:53 2025

@author: garrett
"""

import torch
import torch.nn as nn
import torchvision 
import torchvision.datasets as data
import torch.utils.data.dataloader as dataloader
from torchvision import transforms
from torch.distributions.normal import Normal
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch.optim as optim
import numpy as np
from torchvision.datasets import MNIST

import ResNet2D as resnet
import tsupport
from tsupport import random_crop_with_coords as rcwc
from tsupport import CatVsDogDataset
from ReinLearn import reinforcement_learning as rl


torch.cuda.empty_cache()
device = torch.device("cuda")
cords = torch.tensor(1 * [[-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0]]).to(device)
batch = 1
epochs = 1
alpha = torch.tensor(0.00001).to(device)
lr = torch.tensor(0.0000001).to(device)
sr = torch.tensor(0.01).to(device)
n_classes = 10
norm = Normal(0, 1)
crop_size = torch.tensor(7)
kern = (3,3)
nodes = (crop_size-kern[0])+1
transf = transforms.Compose([
    transforms.ToTensor(),
])

eyeL = norm.sample((32, 8)).to(device)
# trainer = CatVsDogDataset(root=r'/home/garrett/Desktop/Tracker/data', split='trainval', transform=transf, download=False)
# tester = CatVsDogDataset(root=r'/home/garrett/Desktop/Tracker/data', split='test', transform=transf, download=False)
# trainset = data.OxfordIIITPet(root=r'/home/garrett/Desktop/Tracker/data', split='trainval', transform=transf, download=False)

# testset = data.OxfordIIITPet(root=r'/home/garrett/Desktop/Tracker/data', split='test', transform=transf, download=False)


traindata = MNIST('/home/garrett/Desktop/Probabilistic-main/data', train=True, download=False, transform=transf)
testdata = MNIST('/home/garrett/Desktop/Probabilistic-main/data', train=False, download=False, transform=transf)

train_loader = dataloader.DataLoader(traindata, batch_size = batch, shuffle = True)
test_loader = dataloader.DataLoader(testdata, batch_size = batch, shuffle = True)

net = resnet.nolatbuild( in_channels=1, n_classes = n_classes).to(device)
relu = torch.nn.ReLU()
optimizer = optim.Adam(net.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()

saccades = torch.zeros((1, len(train_loader)))
for epoch in range(epochs):
    for i, data in enumerate(train_loader):
        no_choice = True
        evi = torch.zeros((1, n_classes)).to(device)
        im, lab = data
        im = im.to(device)
        lab = lab.to(device)
        
        cropt, w, h = tsupport.crop_image(im= im, crop_size=crop_size, random=True)
        w = torch.tensor(w).to(device)
        h = torch.tensor(h).to(device)
        
        rect = patches.Rectangle((h.to('cpu'),w.to('cpu')), crop_size, crop_size, linewidth=2, edgecolor='red')
        if i % 1000 == 0:  
            plt.imshow(im[0, 0, :, :].detach().to('cpu'))
            plt.gca().add_patch(rect)
            plt.show()
        iii = 0
        r_act = torch.empty((0, 32)).to(device)
        ves = torch.empty((0, 8)).to(device)
        while no_choice:
            iii += 1
            out, x2 = net(cropt.to(device))
            vect = relu(x2 @ eyeL)

            r_act = torch.concatenate([r_act, x2.detach()], axis=0)
            ves = torch.concatenate([ves, vect.detach()], axis=0)
            c = torch.argmax(out)
            card = torch.argmax(vect)
            
            evi += out * sr * iii
            if (evi >= 1).any():
                no_choice == False
                saccades[0, i] = iii
                labeled = torch.argmax(evi)
                break;

            else:
                
                a, b = cords[card]
                w += b
                h += a
                
                if w < 0:
                    w = torch.tensor(0).to(device)
                elif w > im.shape[3]-crop_size:
                    w = torch.tensor(im.shape[3]-crop_size).to(device)
                 
                if h < 0:
                    h = torch.tensor(0).to(device)
                elif h > im.shape[2]-crop_size:
                    h = torch.tensor(im.shape[2]-crop_size).to(device)
                    
                cropt = tsupport.crop_image(im= im, crop_size=crop_size, im_x=h, im_y=w, random=False)
        if i % 1000 == 0:  
            rect = patches.Rectangle((h.to('cpu'),w.to('cpu')), crop_size, crop_size, linewidth=2, edgecolor='red')
            plt.imshow(im[0, 0, :, :].detach().to('cpu'))
            plt.gca().add_patch(rect)
            plt.show()
                    
                
        loss = loss_fn(evi, lab)  
        loss.backward()
        
        optimizer.step()
        w_new = rl(alpha, loss, r_act, ves)
        eyeL += w_new
 

with torch.no_grad():  
    total = 0 
    correct = 0    
    
    for i, data in enumerate(test_loader):
        no_choice = True
        evi = torch.zeros((1, n_classes)).to(device)
        im, lab = data
        im = im.to(device)
        lab = lab.to(device)
        
        cropt, w, h = tsupport.crop_image(im= im, crop_size=crop_size, random=True)
        w = torch.tensor(w).to(device)
        h = torch.tensor(h).to(device)
        # rect = patches.Rectangle((h.to('cpu'),w), crop_size, crop_size, linewidth=2, edgecolor='red')
        # if i % 100 == 0:
        #     plt.imshow(im[0, 0, :, :].detach())
        #     plt.gca().add_patch(rect)
        #     plt.show()
        iii = 0
        r_act = torch.empty((0, 32)).to(device)
        ves = torch.empty((0, 8)).to(device)
        while no_choice:
            iii += 1
            out, x2 = net(cropt)
            vect = relu(x2 @ eyeL)
    
            r_act = torch.concatenate([r_act, x2.detach()], axis=0)
            ves = torch.concatenate([ves, vect.detach()], axis=0)
            c = torch.argmax(out)
            card = torch.argmax(vect)
           
            evi += out * sr * iii
            if (evi >= 1).any():
                no_choice == False
                saccades[0, i] = iii
                labeled = torch.argmax(evi)
                break;
            else:
                a, b = cords[card]
                w += b
                h += a
                
                if w < 0:
                    w = torch.tensor(0).to(device)
                elif w > im.shape[3]-crop_size:
                    w = torch.tensor(im.shape[3]-crop_size).to(device)
                 
                if h < 0:
                    h = torch.tensor(0).to(device)
                elif h > im.shape[2]-crop_size:
                    h = torch.tensor(im.shape[2]-crop_size).to(device)
                    
                cropt = tsupport.crop_image(im= im, crop_size=crop_size, im_x=h, im_y=w, random=False)
        # if i % 100 == 0:        
        #     rect = patches.Rectangle((h,w), crop_size, crop_size, linewidth=2, edgecolor='red')     
            
        #     plt.imshow(im[0, 0, :, :].detach())
        #     plt.gca().add_patch(rect)
        #     plt.show()
    
        correct += (labeled == lab).sum().item()
        total += lab.size(0)
                
            
            
print(correct/total)