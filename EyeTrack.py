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
from ReinLearn import search_learning
from MI_MNIST import MI_MNIST
from generators import generator, gaussian_corr_matrix
import condcova as cc
import time
t0 = time.time()
tc = time.time()

torch.cuda.empty_cache()
device = torch.device("cuda")
cords = torch.tensor([[-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0]]).to(device)
cords *= 1
batch = 8
epochs = 1

lr = torch.tensor(10**-4).to(device)
sr = torch.tensor(0.001).to(device)
n_classes = 8
n_classes2 = 10
norm = Normal(0, 1)
crop_size = torch.tensor(7)
kern = (3,3)
nodes = (crop_size-kern[0])+1
transf = transforms.Compose([
    transforms.ToTensor(),
])

eyeL = norm.sample((32, 8)).to(device)


mi_mnist = MI_MNIST(path='/home/garrett/Desktop/Tracker/data/MNIST_MI')   
testdata = MNIST('/home/garrett/Desktop/Probabilistic-main/data', train=False, download=False, transform=transf)

train_loader = dataloader.DataLoader(mi_mnist, batch_size=batch, shuffle=True)
test_loader = dataloader.DataLoader(testdata, batch_size = batch, shuffle = True)

net = resnet.nolatbuild( in_channels=1, n_classes = n_classes).to(device)
optimizer = optim.Adam(net.parameters(), lr=lr)
loss_fn = search_learning(lr = 10 ** -3, kern_size = kern)

genny = generator().to(device)
criterion = nn.L1Loss()
optimizerG = optim.Adam(genny.parameters(), lr =10 ** -3)
flatter = nn.Flatten()

cater = resnet.nolatbuild_class(in_channels=1, n_classes=10).to(device)
loss_class = nn.CrossEntropyLoss()
optimizerC = optim.Adam(cater.parameters(), lr = 10 ** -3)

saccades = torch.zeros((1, len(train_loader)))
losses = []
xxx = torch.zeros(1000, 1, 28, 28).to(device)
aaa = 0

sw = 8
sh = 8
torch.autograd.set_detect_anomaly(True)
for epoch in range(epochs):
    for i, data in enumerate(train_loader):


        im = data['image'].to(device)     # (B, 1, 28, 28)
        lab = data['label'].to(device)    # (B, ...)
        mi_map = data['mi_map'].to(device)
        
        lab = data['label']
        lab = lab.to(device)
        mi_map = data['image']
        
        
        
        B = im.shape[0]
        w = torch.full((B,), sw, device=device)
        h = torch.full((B,), sh, device=device)
        
        cropt = tsupport.crop_batch(im, w, h, crop_size)
        no_choice = torch.ones(B, dtype=torch.bool, device=device)
        iii = torch.zeros(B, device=device)
        evi = torch.zeros((B, n_classes2), device=device)
        sequence = []

        r_act = torch.empty((0, 32)).to(device)
        ves = torch.empty((0, 8)).to(device)
        while no_choice.any():

            iii[no_choice] += 1
        
            out, x2, x0 = net(cropt)      # out: (B, n_classes)
            out2 = x2 @ torch.ones((32, 10), device=device)
        
            c = torch.argmax(out, dim=1) # (B,)
        
            evi += out2 * sr * iii.unsqueeze(1)
        
            # store sequence
            sequence.append(
                torch.stack([c, w, h], dim=1)
            )
        
            done = iii > 10
            no_choice &= ~done
        
            if not no_choice.any():
                break
        
            # movement
            delta = cords[c]             # (B, 2)
            h = h + delta[:, 0]
            w = w + delta[:, 1]
        
            # clamp
            h = h.clamp(0, im.shape[2] - crop_size)
            w = w.clamp(0, im.shape[3] - crop_size)
        
            cropt = tsupport.crop_batch(im, w, h, crop_size)
                        

        
        sequence = torch.stack(sequence, dim=1) 
        area = sequence[:, 1:]
        suniq = torch.unique(area, dim=0)
        gain = 1/len(suniq)
        
        
        loss = gain*torch.sum(loss_fn(sequence, mi_map.to(device)))
        loss.backward(retain_graph=True)
        # optimizer.step()
        # losses.append(loss.clone().detach().cpu())
        
        
        xim, x0 = genny(im)
        lossG = criterion(xim, im)
        lossG.backward(retain_graph=True)
        optimizerG.step()
        
        xims = xim.clone().detach()
        cout = cater(xims)
        
        loss_c = loss_class(cout, torch.argmax(lab, dim=1))
        loss_alpha = loss_c * loss
        loss_alpha.backward(retain_graph = True)
        # loss_c.backward(retain_graph=True)
        optimizer.step()
        optimizerC.step()
        
        
        ch = torch.argmax(cout)
        labi = torch.argmax(lab)
        
        
        
        aaa += 1
        if aaa > 499:
            ax2 = plt.subplot(221)
            ax2.imshow(mi_map[0, :, :].clone().detach().to('cpu'))
            ax2.plot(sequence[:, 1], sequence[:, 2])
            ax2.plot(sequence[0, 1], sequence[0, 2], c= (0,0,0), marker='o')
            ax2.plot(sequence[-1, 1], sequence[-1, 2], c=(1,0,0), marker='*')
            # plt.gca().add_patch(rect)
            plt.title('After epoch: {}, trial: {}'.format(epoch, i))
            # plt.show()
            
            ax21 = plt.subplot(222)
            inni_im = int_im[ :, :].clone().detach()
            ax21.imshow(inni_im.cpu().numpy())
            plt.title('Time of epoch: {:.2f}'.format(time.time()-tc))
            tc = time.time()
            
            ax22 = plt.subplot(224)
            ximmi = xim[0, 0, :, :].clone().detach()
            ax22.imshow(ximmi.cpu().numpy())
            
            
            ax23= plt.subplot(223)
            ax23.imshow(genny.lats[0:27, 0:27].clone().detach().cpu())
            plt.title('True class {}, guessed class {}'.format(labi, ch))
            plt.show()
            aaa = 0
        
        
            
        #     plt.show()
        #     del sequence
            
        if i != len(train_loader)-1 :
                del sequence
            
    ax2 = plt.subplot(221)
    ax2.imshow(mi_map[0, :, :].detach().to('cpu'))
    ax2.plot(sequence[:, 1], sequence[:, 2])
    ax2.plot(sequence[0, 1], sequence[0, 2], c= (0,0,0), marker='o')
    ax2.plot(sequence[-1, 1], sequence[-1, 2], c=(1,0,0), marker='*')
    # plt.gca().add_patch(rect)
    plt.title('End of epoch: {}'.format(epoch))
    # plt.show()
    
    ax21 = plt.subplot(222)
    inni_im = int_im[ :, :].clone().detach()
    ax21.imshow(inni_im.cpu().numpy())
    plt.title('Time of epoch: {:.2f}'.format(time.time()-tc))
    tc = time.time()
    
    ax22 = plt.subplot(224)
    ximmi = xim[0, 0, :, :].clone().detach()
    ax22.imshow(ximmi.cpu().numpy())
    
    
    ax23= plt.subplot(223)
    ax23.imshow(genny.lats.clone().detach().cpu())
    
    plt.show()
        

print(time.time()-t0)          
 

# with torch.no_grad():  
#     total = 0 
#     correct = 0    
    
#     for i, data in enumerate(test_loader):
#         no_choice = True
#         evi = torch.zeros((1, n_classes)).to(device)
#         im, lab = data
#         im = im.to(device)
#         lab = lab.to(device)
        
#         cropt = tsupport.crop_image(im= im, im_x=11, im_y=11, crop_size=crop_size, random=False)
#         w = torch.tensor(11).to(device)
#         h = torch.tensor(11).to(device)

#         iii = 0
#         r_act = torch.empty((0, 32)).to(device)
#         ves = torch.empty((0, 8)).to(device)
#         while no_choice:
#             iii += 1
#             out, x2 = net(cropt)
#             vect = relu(x2 @ eyeL)
    
#             r_act = torch.concatenate([r_act, x2.detach()], axis=0)
#             ves = torch.concatenate([ves, vect.detach()], axis=0)
#             c = torch.argmax(out)
#             card = torch.argmax(vect)
           
#             evi += out * sr * iii
#             if (evi >= 1).any() or iii == 10:
#                 no_choice == False
#                 saccades[0, i] = iii
#                 labeled = torch.argmax(evi)
#                 break;
#             else:
#                 a, b = cords[card]
#                 w += b
#                 h += a
                
#                 if w < 0:
#                     w = torch.tensor(0).to(device)
#                 elif w > im.shape[3]-crop_size:
#                     w = torch.tensor(im.shape[3]-crop_size).to(device)
                 
#                 if h < 0:
#                     h = torch.tensor(0).to(device)
#                 elif h > im.shape[2]-crop_size:
#                     h = torch.tensor(im.shape[2]-crop_size).to(device)
                    
#                 cropt = tsupport.crop_image(im= im, crop_size=crop_size, im_x=h, im_y=w, random=False)
#         # if i % 100 == 0:        
#         #     rect = patches.Rectangle((h,w), crop_size, crop_size, linewidth=2, edgecolor='red')     
            
#         #     plt.imshow(im[0, 0, :, :].detach())
#         #     plt.gca().add_patch(rect)
#         #     plt.show()
    
#         correct += (labeled == lab).sum().item()
#         total += lab.size(0)
                
            
            
# print(correct/total)