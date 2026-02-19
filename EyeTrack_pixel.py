#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 14 09:52:52 2026

@author: garrett
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EyeTrack_better_predictor.py 
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.utils.data.dataloader as dataloader
from torch.distributions.normal import Normal
from torchvision import transforms
from torchvision.datasets import MNIST

import ResNet2D_batch as resnet
import tsupport
from ReinLearn import search_learning
from MI_MNIST import MI_MNIST
from generators_batch import generator

# ---------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------
t0 = time.time()
tc = time.time()

torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cords = torch.tensor(
    [[-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0]],
    device=device,
    dtype=torch.long,
)

cords *= 3

batch = 8
epochs = 5

lr = 1e-3
sr = 0.001

n_actions = 2
n_classes = 10

norm = Normal(0, 1)

crop_size = 7
kern = (3, 3)

transf = transforms.Compose([transforms.ToTensor()])

# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------
mi_mnist = MI_MNIST(path='/home/garrett/Desktop/Tracker/data/MNIST_MI')
testdata = MNIST('/home/garrett/Desktop/Probabilistic-main/data', train=False, download=False, transform=transf)

train_loader = dataloader.DataLoader(mi_mnist, batch_size=batch, shuffle=True)
test_loader = dataloader.DataLoader(testdata, batch_size=batch, shuffle=True)

# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------
net = resnet.nolatbuild(in_channels=1, n_classes=n_actions).to(device)
optimizer = optim.Adam(net.parameters(), lr=lr)
loss_fn = search_learning(lr=1e-3, kern_size=kern)

# Generator (takes full images: (B,1,28,28))
genny = generator().to(device)
criterionG = nn.L1Loss()
optimizerG = optim.Adam(genny.parameters(), lr=1e-3)

# Classifier (takes generated images: (B,1,28,28))
cater = resnet.nolatbuild_class(in_channels=1, n_classes=n_classes).to(device)
loss_class = nn.CrossEntropyLoss()
optimizerC = optim.Adam(cater.parameters(), lr=1e-3)

# ---------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------
sw, sh = 8, 8  # initial w/h (top-left coords)
torch.autograd.set_detect_anomaly(True)

losses = []
aaa = 0

for epoch in range(epochs):
    for i, batch_data in enumerate(train_loader):

        # ----------------------------
        # Load batch (MI_MNIST returns image and mi_map without channel dim)
        # ----------------------------
        im = batch_data['image'].to(device)     # (B, H, W) or (B, 1, H, W)
        mi_map = batch_data['image'].to(device)  # (B, H, W)
        lab = batch_data['label'].to(device)    # (B, 10) one-hot float
        int_im = norm.sample((batch, 28, 28)).to(device)
        if im.dim() == 3:
            im = im.unsqueeze(1)  # (B, 1, H, W)

        B, C, H, W = im.shape

        # ----------------------------
        # Initialize crop coords (batched)
        # ----------------------------
        w = torch.full((B,), sw, device=device, dtype=torch.long)
        h = torch.full((B,), sh, device=device, dtype=torch.long)

        cropt = tsupport.crop_batch(im, w, h, crop_size)  # (B,1,7,7)

        no_choice = torch.ones(B, dtype=torch.bool, device=device)
        iii = torch.zeros(B, device=device, dtype=torch.float32)

        evi = torch.zeros((B, n_classes), device=device, dtype=torch.float32)
        sequence = []

        # ----------------------------
        # Saccade loop
        # ----------------------------
        while no_choice.any():
            iii[no_choice] += 1.0

            out, x2, x0 = net(cropt)                # out: (B, n_actions)
            out2 = x2 @ torch.ones((32, n_classes), device=device)  # (B, 10)

            c = torch.argmax(out, dim=1).to(torch.long)  # (B,)

            evi = evi + out2 * sr * iii.unsqueeze(1)
            
                        
            for b in range(batch):
                if not no_choice[b]:
                    continue
            
                hb = h[b].item()
                wb = w[b].item()
            
                int_im[b,
                       hb:hb+crop_size,
                       wb:wb+crop_size] +=  x0[b, 0]

            # store [action, w, h] as longs for consistent indexing
            sequence.append(torch.stack([c, w, h], dim=1))

            done = iii > 5
            no_choice = no_choice & (~done)

            if not no_choice.any():
                break

            # Move each sample based on its chosen action
            h = torch.tensor( torch.round( 28 * out[:, 0]), dtype=torch.long)
            w = torch.tensor( torch.round(28 * out[:, 1]), dtype=torch.long)

            # Clamp into valid crop range
            h = h.clamp(0, H - crop_size)
            w = w.clamp(0, W - crop_size)

            cropt = tsupport.crop_batch(im, w, h, crop_size)

        # (B, T, 3)
        sequence = torch.stack(sequence, dim=1)

        # ----------------------------
        # Gain / search loss
        # ----------------------------
        # NOTE: this preserves your original structure, but now everything is batched/device-safe.
        # area = sequence[:, 1:]  # (B, T, 2)
        # suniq = torch.unique(area, dim=0)
        # gain = 1.0 / max(len(suniq), 1)

        # loss_search = gain * torch.sum(loss_fn(sequence, mi_map))
        # loss_search.backward(retain_graph=True)
        loss_search = torch.sum(loss_fn(sequence, mi_map))
        # ----------------------------
        # Generator loss (batch-safe): feed full batch, no extra unsqueeze
        # ----------------------------
        xim, x0_full = genny(int_im.unsqueeze(1))          # xim: (B,1,28,28)
        lossG = criterionG(xim, im)


        # ----------------------------
        # Classifier loss (batch-safe)
        # ----------------------------
        logits = cater(xim.detach())      # (B,10) logits (no softmax)
        targets = torch.argmax(lab, dim=1)  # (B,) long
        loss_c = loss_class(logits, targets)

        # preserve your coupling between losses
        loss_alpha = lossG * loss_search
        loss_alpha.backward(retain_graph=True)
        
        lossG.backward(retain_graph=True)
        optimizerG.step()
        optimizerG.zero_grad(set_to_none=True)
        
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        optimizerC.step()
        optimizerC.zero_grad(set_to_none=True)

        losses.append(float(loss_search.detach().cpu()))

        # ----------------------------
        # Debug plots
        # ----------------------------
        aaa += 1
        if aaa > 499:
            # show first sample only
            b0 = 0

            ax2 = plt.subplot(221)
            ax2.imshow(im[b0, 0].detach().cpu().numpy())
            seq0 = sequence[b0].detach().cpu()
            ax2.plot(seq0[:, 1], seq0[:, 2])
            ax2.plot(seq0[0, 1], seq0[0, 2], c=(0, 0, 0), marker='o')
            ax2.plot(seq0[-1, 1], seq0[-1, 2], c=(1, 0, 0), marker='*')
            plt.title(f'After epoch: {epoch}, trial: {i}')

            ax21 = plt.subplot(222)
            ax21.imshow(int_im[b0].detach().cpu())
            plt.title('Time of epoch: {:.2f}'.format(time.time() - tc))
            tc = time.time()

            ax22 = plt.subplot(224)
            ax22.imshow(xim[b0, 0].detach().cpu().numpy())

            ax23 = plt.subplot(223)
            ax23.imshow(genny.lats[0:27, 0:27].detach().cpu())
            pred = torch.argmax(logits[b0]).item()
            true = targets[b0].item()
            plt.title(f'True class {true}, guessed class {pred}')
            plt.show()
            aaa = 0

        if i != len(train_loader) - 1:
            del sequence

    print(f"Epoch {epoch} finished. Elapsed: {time.time() - t0:.2f}s")

print(time.time() - t0)
