import torch
import torchvision
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torch.optim as optim

from tqdm import trange
import wandb

import numpy as np
import os
import random
import ipdb

from mixup import to_one_hot

def gradmix(x, y, grad):
    '''Returns mixed inputs, pairs of targets, and lambda'''

    batch_size, c, w, h = np.array(x.size())

    index = torch.randperm(batch_size).cuda()

    mixed_y = [y, y[index]]
    M = (grad) / (grad + grad[index,:])

    grad_mean = grad.mean(dim=(1,2,3))
    _mixed_lam = grad_mean / (grad_mean + grad_mean[index])
    mixed_lam = [_mixed_lam.detach(), 1- _mixed_lam.detach()]

    mixed_x = M*x + (1-M)*x[index]

    return mixed_x.detach(), mixed_y, mixed_lam

def normalize_grad(grad):
    grad_min = grad.amin(dim=[1,2,3],keepdim=True)
    grad -= grad_min
    grad_max = grad.amax(dim=[1,2,3],keepdim=True)
    grad /= grad_max
    return grad

def gradmix_v2(x, y, grad, stride=10, debug=False):
    '''Returns mixed inputs, pairs of targets, and lambda'''

    batch_size, c, w, h = np.array(x.size())
    if debug:
        # index = torch.arange(start=batch_size, end=0, step=-1)-1
        index = torch.tensor([1,4,3,0,2]).cuda()
        # index = index.cuda()
    else:
        index = torch.randperm(batch_size).cuda()

    mixed_y = [y, y[index]]

    max_criteria = torch.zeros([batch_size]).cuda()
    grad_1, grad_2 = grad, grad[index, :]
    mixed_x = torch.zeros_like(x).cuda()
    _mixed_lam = torch.zeros([batch_size]).cuda()

    # normalized_grad_1 = grad_1/grad_1.amax(dim=[1,2,3],keepdim=True)
    normalized_grad_1 = normalize_grad(grad_1)
    # padded_grad_1 = torch.nn.functional.pad(grad_1, [w,w,w,w], mode='constant', value=0.0)
    padded_normalized_grad_1 = torch.nn.functional.pad(normalized_grad_1, [w,w,w,w], mode='constant', value=0.0)
    padded_x_1 = torch.nn.functional.pad(x, [w,w,w,w], mode='constant', value=0.0)

    for i in range(0,w,stride):
        for j in range(0,w,stride):
            # normalized_grad_2 = grad_2/grad_2.amax(dim=[1,2,3],keepdim=True)
            normalized_grad_2 = normalize_grad(grad_2)
            # padded_grad_2 = torch.nn.functional.pad(grad_2, [w-j,w+j,0+i,2*w-i], mode='constant', value=0.0)
            padded_normalized_grad_2 = torch.nn.functional.pad(normalized_grad_2, [w-j,w+j,0+i,2*w-i], mode='constant', value=0.0)
            padded_x_2 = torch.nn.functional.pad(x[index,:], [w-j,w+j,0+i,2*w-i], mode='constant', value=0.0)
            
            M = padded_normalized_grad_1 / (padded_normalized_grad_1+padded_normalized_grad_2+1e-6)
            lambbda = M[:,:,w:2*w,w:2*w].mean(dim=[1,2,3])
            M_adjusted = M.expand(-1,3,-1,-1)
            
            current_saliency = (padded_normalized_grad_1 * M)[:,:,w:2*w,w:2*w] + (padded_normalized_grad_2 * (1-M))[:,:,w:2*w,w:2*w]
            
            criteria = current_saliency.sum(dim=[1,2,3])
            update_needed = ((criteria - max_criteria)>0)
            
            if update_needed.sum() >0:
                mixed_x[update_needed,:,:,:] = (padded_x_1 * M_adjusted)[update_needed,:,w:2*w,w:2*w] + (padded_x_2 * (1-M_adjusted))[update_needed,:,w:2*w,w:2*w]
                _mixed_lam[update_needed] = lambbda[update_needed]
                max_criteria[update_needed] = criteria[update_needed]

    mixed_lam = [_mixed_lam.detach(), 1- _mixed_lam.detach()]
    
    # del max_criteria, _mixed_lam, index, M, M_adjusted
    # del padded_normalized_grad_1, padded_normalized_grad_2, normalized_grad_1, normalized_grad_2, padded_x_1, padded_x_2

    grad_2, grad_1 = grad, grad[index, :]

    # normalized_grad_1 = grad_1/grad_1.amax(dim=[1,2,3],keepdim=True)
    normalized_grad_1 = normalize_grad(grad_1)
    padded_normalized_grad_1 = torch.nn.functional.pad(normalized_grad_1, [w,w,w,w], mode='constant', value=0.0)
    padded_x_1 = torch.nn.functional.pad(x[index,:], [w,w,w,w], mode='constant', value=0.0)

    for i in range(0,w,stride):
        for j in range(0,w,stride):
            # normalized_grad_2 = grad_2/grad_2.amax(dim=[1,2,3],keepdim=True)
            normalized_grad_2 = normalize_grad(grad_2)
            padded_normalized_grad_2 = torch.nn.functional.pad(normalized_grad_2, [w-j,w+j,0+i,2*w-i], mode='constant', value=0.0)
            padded_x_2 = torch.nn.functional.pad(x, [w-j,w+j,0+i,2*w-i], mode='constant', value=0.0)

            M = padded_normalized_grad_1 / (padded_normalized_grad_1+padded_normalized_grad_2+1e-6)
            lambbda = M[:,:,w:2*w,w:2*w].mean(dim=[1,2,3])
            M_adjusted = M.expand(-1,3,-1,-1)

            current_saliency = (padded_normalized_grad_1 * M)[:,:,w:2*w,w:2*w] + (padded_normalized_grad_2 * (1-M))[:,:,w:2*w,w:2*w]

            criteria = current_saliency.sum(dim=[1,2,3])
            update_needed = ((criteria - max_criteria)>0)

            if update_needed.sum() >0:
                mixed_x[update_needed,:,:,:] = (padded_x_1 * M_adjusted)[update_needed,:,w:2*w,w:2*w] + (padded_x_2 * (1-M_adjusted))[update_needed,:,w:2*w,w:2*w]
                _mixed_lam[update_needed] = lambbda[update_needed]
                max_criteria[update_needed] = criteria[update_needed]

    return mixed_x.detach(), mixed_y, mixed_lam

def reweighted_lam(mixed_y, mixed_lam, num_classes):
    y0 = to_one_hot(mixed_y[0], num_classes)
    y1 = to_one_hot(mixed_y[1], num_classes)

    return mixed_lam[0].unsqueeze(1)*y0 + mixed_lam[1].unsqueeze(1)*y1
