#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 21 12:03:41 2025

@author: garrett
"""

import torch

def samp_cov(X, unbiased=True):
    n = X.shape[2]
    X = X[:, 0, :, :]
    X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
    mean = torch.mean(X, dim = 0)
    Xc = X - mean
    S = (Xc.T @ Xc)/ (n - 1 if unbiased else n)
    return S

def project_span(alpha, delta):
    amin = torch.min(alpha)
    amax = torch.max(alpha)
    
    if amax - amin <=delta:
        return delta
    
    sorteda = torch.sort(alpha)[0]
    n = len(alpha)
    best_proj = None
    min_sq = torch.inf
    for i in range(n):
        low = sorteda[i]
        high = low + delta
        proj = torch.clip(alpha, low, high)
        sq = torch.sum((proj - alpha) ** 2)
        if sq < min_sq:
            min_sq = sq
            best_proj = proj
    
    return best_proj

def cond_reg_cov(X, kappa_max):
    S = samp_cov(X)
    eigvals, eigvecs = torch.linalg.eigh(S)
    eigvals = torch.clip(eigvals, 1e-12, None)
    alpha = torch.log(eigvals)
    alpha_proj = project_span(alpha, torch.log(kappa_max))
    mu = torch.exp(alpha_proj)
    S_new = eigvecs @ torch.diag(mu) @ eigvecs.T 
    return S_new

def weight_expansion(weights, node_i = None):
    if node_i == None:
        node_i = weights.shape[-1]-1
    new_nodes = torch.zeros((weights.shape[0], weights.shape[1], 1))

    U, S, Vh = torch.linalg.svd(weights)
    idx = torch.argmax(S)
    
    A = S[0, idx] * U[0, idx, :]
    A /= A.std()
    
    weights_mean = weights.mean(dim=1)
    weights_std = weights.std(dim = 1)
    weights_norm = (weights - weights_mean)/weights_std
    
    A_mean = A.mean()
    A_std = A.std()
    A_norm = (A - A_mean)/A_std
    
    c = (weights_norm * A_norm.unsqueeze(1)).mean(dim=1)
    node_i = torch.argmax(c)
    print('and the max weight node is {}'.format(node_i))
    a = weights[:, :, 0:node_i+1]
    # if a.dim() == 2:
    #     a = a.unsqueeze(dim=1)
    b = weights[:, :, node_i+1:]
    # if b.dim() == 2:
    #     b = b.unsqueeze(dim=1)
    weight_up = torch.cat([a, new_nodes, b], dim=2)
    weight_up[:, :, node_i+1] = A
    weight_up /= weight_up.max()
    nodes = weight_up.shape[0]
    return weight_up, nodes, node_i

def cov_expansion(weights, node_i = None):
    if node_i == None:
        node_i = weights.shape[-1]-1
    node_i += 1

    weight_up = torch.zeros(weights.shape[1]+1, weights.shape[1]+1)
    
    weight_up[:node_i, :node_i] = weights[:node_i, :node_i]
    weight_up[:node_i, node_i+1:] = weights[:node_i, node_i:]
    weight_up[node_i+1:, :node_i] = weights[node_i:, :node_i]
    weight_up[node_i+1:, node_i+1:] = weights[node_i:, node_i:]

    U, S, Vh = torch.linalg.svd(weights)
    idx = torch.argmax(S)
    print('and the max Cov node is {}'.format(idx))
    A = S[idx] * U[idx, :]
    A /= A.std()
    weight_up[node_i, 0:node_i] += A[:node_i]
    weight_up[node_i, node_i+1:] += A[node_i:]
    weight_up[0:node_i, node_i] += A[:node_i]
    weight_up[node_i+1:, node_i] += A[node_i:]
    weight_up[node_i, node_i] = torch.std(A)
    # weight_up /= weight_up.max()
    nodes = weight_up.shape[0]
    return weight_up, nodes