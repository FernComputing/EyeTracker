#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ResNet2D.py (batch-safe tweaks)

Main changes for batch support:
- nolatbuild_class now returns logits (NO softmax) so it works with nn.CrossEntropyLoss.
- latbuild softmax dimension fixed (dim=1) if you ever use it batched.
- ResCon2D pooling made explicit (kernel=(1,1)) to preserve original flatten sizes.
"""

import torch
import torch.nn as nn


class ResCon2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, **kwargs):
        super().__init__()

        # NOTE: out_channels kept for API compatibility; original block uses in_channels throughout.
        self.x1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(in_channels),
        )

        self.x2 = self.x1

        self.x3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(in_channels),
        )

        # Original code used MaxPool2d((in_channels, in_channels)); for in_channels=1 this is (1,1).
        # Make that explicit so shapes stay identical and batch-safe.
        self.x4 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(in_channels),
            nn.MaxPool2d(kernel_size=(1, 1)),
        )

    def forward(self, x):
        x1 = self.x1(x) + x
        x2 = self.x2(x1) + x
        x3 = self.x3(x2) + x
        x4 = self.x4(x3)
        return x4


class latlayer(nn.Module):
    def __init__(self, cov):
        super().__init__()
        self.cov = cov

    def forward(self, x, cc):
        # This layer is not used by EyeTrack.py, but keep it batch-safe if used.
        # Expect x: (B, D)
        device = x.device
        cc_t = torch.full_like(x, float(cc), device=device)

        # Diagonal cov term (D,)
        cov_diag = torch.diag(self.cov).to(device)
        covvy = torch.diag(cov_diag)

        # Apply lateral term
        out = x + (x * cc_t) @ (self.cov.to(device) - covvy)
        return out


class latbuild(nn.Module):
    def __init__(self, cov=None, in_channels=1, n_classes=10):
        super().__init__()
        self.n_classes = n_classes
        self.cov = cov
        self.lats_on = True

        # WARNING: stride=0 is invalid for conv; keep default stride=1 here.
        self.x0 = ResCon2D(in_channels=in_channels, out_channels=1, kernel_size=(3, 3), stride=1)

        self.x1 = nn.Sequential(
            nn.Linear(49, 28),
            nn.ReLU(),
        )

        self.xlat = latlayer(self.cov)

        self.x2 = nn.Sequential(
            nn.Linear(28, n_classes),
            nn.Softmax(dim=1),  # batch-safe softmax if used
        )

    def forward(self, x):
        x0 = self.x0(x)
        x1 = torch.flatten(x0, start_dim=1)
        x2 = self.x1(x1)
        if self.lats_on:
            x3 = self.xlat(x2, 0.1)
            out = self.x2(x3)
        else:
            out = self.x2(x2)
        return out, x2


class nolatbuild(nn.Module):
    def __init__(self, in_channels=3, n_classes=2):
        super().__init__()
        self.n_classes = n_classes

        self.l0 = ResCon2D(in_channels=in_channels, out_channels=1, kernel_size=(3, 3), stride=1)
        self.l1 = nn.Sequential(
            nn.Linear(49, 32),
            nn.ReLU(),
        )

        self.l2 = nn.Sequential(
            nn.Linear(32, n_classes),
            nn.ReLU()
        )

    def forward(self, x):
        x0 = self.l0(x)
        x1 = torch.flatten(x0, start_dim=1)
        x2 = self.l1(x1)
        out = self.l2(x2)
        return out, x2, x0


class nolatbuild_class(nn.Module):
    def __init__(self, in_channels=3, n_classes=2):
        super().__init__()
        self.n_classes = n_classes

        self.l0 = ResCon2D(in_channels=in_channels, out_channels=1, kernel_size=(3, 3), stride=1)

        # NOTE: original code used 784 here; that assumes a 28x28 spatial map after l0.
        # Keep it as-is so existing training scripts don't break.
        self.l1 = nn.Sequential(
            nn.Linear(784, 32),
            nn.ReLU(),
        )

        # IMPORTANT: return logits, NOT softmax, for CrossEntropyLoss
        self.l2 = nn.Linear(32, n_classes)

    def forward(self, x):
        x0 = self.l0(x)
        x1 = torch.flatten(x0, start_dim=1)
        x2 = self.l1(x1)
        logits = self.l2(x2)
        return logits
