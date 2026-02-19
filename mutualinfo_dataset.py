#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 19 20:46:17 2025

@author: garrett
"""
from MutualInfo_data import compute_pixelwise_mi

import torch
import torch.utils.data.dataloader as dataloader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import torch.nn.functional as F
import h5py
batch = 60000
n_class = 10
mean_maps = torch.zeros((n_class, 28, 28))

transf = transforms.Compose([
    transforms.ToTensor(),
])

traindata = MNIST('/home/garrett/Desktop/Probabilistic-main/data', train=True, download=False, transform=transf)
train_loader = dataloader.DataLoader(traindata, batch_size = batch, shuffle = True)

images, labels = next(iter(train_loader))

# for n in range(n_class):
#     sub_labs = torch.where(labels == n)[0]
#     mean_maps[n, :, :] = torch.mean(images[sub_labs, :, :, :], axis = 0)
#     im_a = images[sub_labs[0], :, :, :]
#     mean_im = mean_maps[n, :, :]
#     a = im_a.squeeze(0).T @ mean_im
    
#     im_b  = images[sub_labs[100], :, :, :]
#     b = im_b.squeeze(0).T @ mean_im
    
#     ax1 = plt.subplot((221))
#     ax1.imshow(a)
    
#     ax2 = plt.subplot((222))
#     ax2.imshow(b)
    
#     ax3 = plt.subplot((223))
#     ax3.imshow(im_a.squeeze(0))
    
#     ax4 = plt.subplot((224))
#     ax4.imshow(im_b.squeeze(0))
#     plt.show()



mi_map = compute_pixelwise_mi(images, labels)
mi_maps =  torch.zeros((n_class, 28, 28))
marg_maps = torch.zeros((n_class, 28, 28))
### Will come back to this idea, averaging just seems easier ###
for n in range(n_class):
    sub_labs = torch.where(labels != n)[0]
    subs = torch.where(labels == n)[0]
    mi_maps[n, :, :] = compute_pixelwise_mi(images[sub_labs, :, :, :], labels[sub_labs])
    marg_maps[n, : , :] = mi_maps[n, :, :] - mi_map
    for j in range(len(subs)):
        im_a = images[subs[j], :, :, :]
        a = im_a.squeeze(0).T @ marg_maps[n, : , :]
        mi_name = ('./data/MNIST_MI/MI{}_{}.h5'.format(n, j))
        with h5py.File(mi_name, 'w') as f:
            f.create_dataset('MI_map', data=a)
            f.create_dataset('image', data = im_a.squeeze(0))
            f.create_dataset('MI_class', data = marg_maps[n,:, :])
            

fi_name = h5py.File('./data/MNIST_MI/MI{}_{}.h5'.format(4, 2345), 'r')
MI_im = fi_name['MI_map'][()]
plt.imshow(MI_im)
plt.show()


mi_name = ('./data/MNIST_MI/MI_all.h5')
f = h5py.File(mi_name, 'w') 
f.create_dataset('MI_map_all', data=mi_map)


    # ax1 = plt.subplot((321))
    # ax1.imshow(mi_maps[n, :, :])
    
    # ax2 = plt.subplot((322))
    # ax2.imshow(marg_maps[n, :, :])
    
    # ax3 = plt.subplot((323))
    # ax3.imshow(im_a.squeeze(0))
    
    # ax4 = plt.subplot((324))
    # ax4.imshow(a)
    
    # ax5 = plt.subplot((325))
    # ax5.imshow(im_b.squeeze(0))
    
    # ax6 = plt.subplot((326))
    # ax6.imshow(b)
    
    # plt.show()
