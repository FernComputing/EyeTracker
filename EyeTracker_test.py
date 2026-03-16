

"""
EyeTrack_noClass.py (batch-safe rewrite)

Key changes:
This experiment, is to try a
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.utils.data.dataloader as dataloader
from torch.utils.data import DataLoader, random_split
from torch.distributions.normal import Normal
from torchvision import transforms
from torchvision.datasets import MNIST
import torchmetrics
from torchmetrics.classification import MulticlassROC
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import ResNet2D_batch as resnet
import tsupport
from ReinLearn import search_learning
from MI_MNIST import MI_MNIST
from generators_batch import generator
from torch.utils.tensorboard import SummaryWriter
import pytorch_msssim
lats_on = True
saccades = 4

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

cords *= 7

batch = 64
epochs = 25
lr = 1e-3
sr = 0.001

n_actions = 8
n_classes = 10

norm = Normal(0, 1)

crop_size = 7
kern = (3, 3)

transf = transforms.Compose([transforms.ToTensor()])

# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------
mi_mnist = MI_MNIST(path='/home/garrett/Desktop/Tracker/data/MNIST_MI')
# Compute split sizes
train_size = int(0.8 * len(mi_mnist))
test_size = (len(mi_mnist) - train_size)
# Perform split


genera = torch.Generator().manual_seed(10923)

train_dataset, test_dataset = random_split(
    mi_mnist,
    [train_size, test_size],
    generator=genera
)

train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True, num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_dataset, batch_size=batch, shuffle=False, num_workers=4, pin_memory=True)

# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------
net = resnet.nolatbuild(in_channels=1, n_classes=n_actions).to(device)
optimizer = optim.Adam(net.parameters(), lr=lr)
reward_fn = search_learning(lr=1e-3, kern_size=kern)
epoch_lossPoli = []
# Generator (takes full images: (B,1,28,28))
genny = generator(lats_on=lats_on).to(device)
criterionG1 = nn.L1Loss()
criterionG = nn.MSELoss()
optimizerG = optim.Adam(genny.parameters(), lr=1e-3)
epoch_lossG = []

# Classifier
cater = resnet.nolatbuild_class(in_channels=1, n_classes=n_classes).to(device)
loss_class = nn.CrossEntropyLoss()
optimizerC = optim.Adam(cater.parameters(), lr=1e-3)
epoch_lossC = []
# ---------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------
sw, sh = 8, 8  # initial w/h (top-left coords)
# torch.autograd.set_detect_anomaly(True)

aaa = 0
net.train()
genny.train()
cater.train()
for epoch in range(epochs):
    running_epoch_loss_G = 0.0
    running_epoch_loss_C = 0.0
    running_epoch_loss_poli = 0.0
    for i, batch_data in enumerate(train_loader):

        # ----------------------------
        # Load batch (MI_MNIST returns image and mi_map without channel dim)
        # ----------------------------
        im = batch_data['image'].to(device)     # (B, H, W) or (B, 1, H, W)
        mi_map = batch_data['image'].to(device)  # (B, H, W)
        lab = batch_data['label'].to(device)    # (B, 10) one-hot float

        if im.dim() == 3:
            im = im.unsqueeze(1)  # (B, 1, H, W)

        B, C, H, W = im.shape
        int_im = norm.sample((B, 28, 28)).to(device)
        # ----------------------------
        # Initialize crop coords (batched)
        # ----------------------------
        w = torch.full((B,), sw, device=device, dtype=torch.long)
        h = torch.full((B,), sh, device=device, dtype=torch.long)
        logps = []
        cropt = tsupport.crop_batch(im, w, h, crop_size)  # (B,1,7,7)

        no_choice = torch.ones(B, dtype=torch.bool, device=device)
        iii = torch.zeros(B, device=device, dtype=torch.float32)

        sequence = []

        # ----------------------------
        # Saccade loop
        # ----------------------------
        for j in range(saccades):
            iii[no_choice] += 1.0

            out, x2, x0 = net(cropt)                # out: (B, n_actions)

            dist = torch.distributions.Categorical(logits=out) 
            c = dist.sample()
            logps.append(dist.log_prob(c))
            int_im = tsupport.add_patch_batch(int_im, x0, h, w, crop_size)
            # int_im /=  int_im.max()
            # store [action, w, h] as longs for consistent indexing
            sequence.append(torch.stack([c, w, h], dim=1))

            # Move each sample based on its chosen action
            delta = cords[c]      # (B, 2) long
            h = h + delta[:, 0]
            w = w + delta[:, 1]

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
        area = sequence[:, 1:]  # (B, T, 2)
        suniq = torch.unique(area, dim=0)
        gain = 1.0 #/ max(len(suniq), 1)

        R = gain * reward_fn(sequence, mi_map)
        baseline = R.mean()
        adv = (R - baseline).detach()
        logp_traj = torch.stack(logps, dim=0).sum(dim=0)  # (B,)
        policy_loss = -(adv * logp_traj).mean()
        policy_loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # loss_search.backward(retain_graph=True)

        # ----------------------------
        # Generator loss (batch-safe): feed full batch, no extra unsqueeze
        # ----------------------------
        xim, x0_full = genny(int_im.unsqueeze(1))  
        lossA = 1 - pytorch_msssim.ssim(xim, im)        # xim: (B,1,28,28)
        lossG1 = criterionG1(xim, im)
        lossG = (lossG1 + criterionG(xim, im) + lossA)
        
        lossG.backward()
        optimizerG.step()
        optimizerG.zero_grad(set_to_none=True)

        # ----------------------------
        # Classifier loss (batch-safe)
        # ----------------------------    # (B,10) logits (no softmax)
        logits = cater(xim.detach())      # (B,10) logits (no softmax)
        choice = torch.argmax(logits, dim=1) 
        targets = torch.argmax(lab, dim=1)  # (B,) long
        loss_c = loss_class(logits, targets)
        loss_c.backward()
        optimizerC.step()
        optimizerC.zero_grad(set_to_none=True)


        running_epoch_loss_G += lossG.item() * im.size(0)
        running_epoch_loss_C += loss_c.item() * im.size(0)
        running_epoch_loss_poli += policy_loss.item() * im.size(0)
                                                        
        
        # ----------------------------
        # Debug plots
        # ----------------------------

        # show first sample only
    b0 = 0

    ax2 = plt.subplot(131)
    ax2.imshow(im[b0, 0].detach().cpu().numpy())
    seq0 = sequence[b0].detach().cpu()
    ax2.plot(seq0[:, 1], seq0[:, 2])
    ax2.plot(seq0[0, 1], seq0[0, 2], c=(0, 0, 0), marker='o')
    ax2.plot(seq0[-1, 1], seq0[-1, 2], c=(1, 0, 0), marker='*')
    plt.title(f'After epoch: {epoch}, trial: {i}')

    ax21 = plt.subplot(132)
    ax21.imshow(int_im[b0].detach().cpu())
    
    tc = time.time()

    ax22 = plt.subplot(133)
    ax22.imshow(xim[b0, 0].detach().cpu().numpy())
    plt.title(f'guessed {choice[0]}, truth was {targets[0]}')
    plt.show()
    aaa = 0
    if i != len(train_loader) - 1:
        del sequence

    print(f"Epoch {epoch} finished. Elapsed: {time.time() - t0:.2f}s")
    epoch_lossG.append(running_epoch_loss_G / len(train_dataset))
    epoch_lossC.append(running_epoch_loss_C / len(train_dataset))
    epoch_lossPoli.append(running_epoch_loss_poli / len(train_dataset))

    print(f"Epoch [{epoch+1}/{epochs}] "
        f"Search Train Loss: {epoch_lossPoli[epoch]:.4f} "
        f"Generator Train Loss: {epoch_lossG[epoch]:.4f}"
        f"Classifier Train Loss: {epoch_lossC[epoch] :.4f}")
    
print(time.time() - t0)

 


net.eval()
genny.eval()
cater.eval()
metrics = MulticlassROC(num_classes=10)
aaa = 0
with torch.no_grad():
     for batch_data in test_loader:

        # ----------------------------
        # Load batch (MI_MNIST returns image and mi_map without channel dim)
        # ----------------------------
        im = batch_data['image'].to(device)     # (B, H, W) or (B, 1, H, W)
        mi_map = batch_data['image'].to(device)  # (B, H, W)
        lab = batch_data['label'].to(device)    # (B, 10) one-hot float

        if im.dim() == 3:
            im = im.unsqueeze(1)  # (B, 1, H, W)

        B, C, H, W = im.shape
        int_im = norm.sample((B, 28, 28)).to(device)
        # ----------------------------
        # Initialize crop coords (batched)
        # ----------------------------
        w = torch.full((B,), sw, device=device, dtype=torch.long)
        h = torch.full((B,), sh, device=device, dtype=torch.long)
        logps = []
        cropt = tsupport.crop_batch(im, w, h, crop_size)  # (B,1,7,7)

        no_choice = torch.ones(B, dtype=torch.bool, device=device)
        iii = torch.zeros(B, device=device, dtype=torch.float32)

        sequence = []

        # ----------------------------
        # Saccade loop
        # ----------------------------
        for j in range(saccades):
            iii[no_choice] += 1.0

            out, x2, x0 = net(cropt)                # out: (B, n_actions)

            dist = torch.distributions.Categorical(logits=out) 
            c = dist.sample()
            logps.append(dist.log_prob(c))
            int_im = tsupport.add_patch_batch(int_im, x0, h, w, crop_size)
 
            # int_im /=  int_im.max()
            # store [action, w, h] as longs for consistent indexing
            sequence.append(torch.stack([c, w, h], dim=1))

            # Move each sample based on its chosen action
            delta = cords[c]      # (B, 2) long
            h = h + delta[:, 0]
            w = w + delta[:, 1]

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
        area = sequence[:, 1:]  # (B, T, 2)
        suniq = torch.unique(area, dim=0)
        gain = 1.0 / max(len(suniq), 1)
    
        # loss_search.backward(retain_graph=True)
    
        # ----------------------------
        # Generator loss (batch-safe): feed full batch, no extra unsqueeze
        # ----------------------------
        xim, x0_full = genny(int_im.unsqueeze(1))          # xim: (B,1,28,28)
        lossG = criterionG(xim, im)

    
        # ----------------------------
        # Classifier loss (batch-safe)
        # ----------------------------    # (B,10) logits (no softmax)
        logits = cater(xim.detach())      # (B,10) logits (no softmax)
        choice = torch.argmax(logits, dim=1) 
        targets = torch.argmax(lab, dim=1) 
        
        metrics.update(torch.softmax(logits, dim=1).cpu(), targets.cpu())
        # ----------------------------
        # Debug plots
        # ----------------------------
        # show first sample only
        b0 = 0
        aaa += 1
        if aaa > len(test_loader)/8:
            ax2 = plt.subplot(131)
            ax2.imshow(im[b0, 0].detach().cpu().numpy())
            seq0 = sequence[b0].detach().cpu()
            ax2.plot(seq0[:, 1], seq0[:, 2])
            ax2.plot(seq0[0, 1], seq0[0, 2], c=(0, 0, 0), marker='o')
            ax2.plot(seq0[-1, 1], seq0[-1, 2], c=(1, 0, 0), marker='*')
            plt.title(f'Test trial: {i}')
    
            ax21 = plt.subplot(132)
            ax21.imshow(int_im[b0].detach().cpu())
            tc = time.time()
    
            ax22 = plt.subplot(133)
            ax22.imshow(xim[b0, 0].detach().cpu().numpy())
            plt.title(f'guessed {choice[0]}, truth was {targets[0]}')
            plt.show()
            aaa = 0

ax2 = plt.subplot(131)
ax2.imshow(im[b0, 0].detach().cpu().numpy())
seq0 = sequence[b0].detach().cpu()
ax2.plot(seq0[:, 1], seq0[:, 2])
ax2.plot(seq0[0, 1], seq0[0, 2], c=(0, 0, 0), marker='o')
ax2.plot(seq0[-1, 1], seq0[-1, 2], c=(1, 0, 0), marker='*')
plt.title(f'After epoch: {epoch}, trial: {i}')

ax21 = plt.subplot(132)
ax21.imshow(int_im[b0].detach().cpu())
plt.title('Time of epoch: {:.2f}'.format(time.time() - tc))
tc = time.time()

ax22 = plt.subplot(133)
ax22.imshow(xim[b0, 0].detach().cpu().numpy())

plt.show()
print('Program complete')


fpr, tpr, thresholds = metrics.compute()
aucs = torch.tensor([
    torch.trapz(tpr[i], fpr[i])
    for i in range(len(fpr))
])

macro_auc = aucs.mean()

for x in [epoch_lossG, epoch_lossC, epoch_lossPoli]:
    plt.plot(range(epochs), x, label = f"{x}")

plt.show()


plt.figure(figsize=(7,6))

for i in range(len(fpr)):
    plt.plot(fpr[i], tpr[i], label=f"Class {i}, AUC {aucs[i]}")

plt.plot([0,1], [0,1], linestyle="--")  # random baseline

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multiclass ROC")
plt.legend()
plt.show()
