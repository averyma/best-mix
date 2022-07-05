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

def normalize_grad(grad, alpha=1, normalization = 'standard'):

    # normalize to [0,1]
    if normalization == 'standard':
        # grad_min = grad.amin(dim=[1,2,3],keepdim=True)
        # grad -= grad_min
        # grad_max = grad.amax(dim=[1,2,3],keepdim=True)
        # grad /= grad_max
        # grad *= np.clip(alpha, a_min=1e-6, a_max=None)
        grad.sub_(grad.amin(dim=[1,2,3],keepdim=True))
        grad.div_(grad.amax(dim=[1,2,3],keepdim=True))
        grad.mul_(np.clip(alpha, a_min=1e-6, a_max=None))
    elif normalization == 'L1':
    # L1 normalization
        # grad /= grad.sum(dim=[1,2,3],keepdim=True)
        # grad *= np.clip(alpha, a_min=1e-6, a_max=None)
        grad.div_(grad.sum(dim=[1,2,3],keepdim=True))
        grad.mul_(np.clip(alpha, a_min=1e-6, a_max=None))
    elif normalization == 'softmax':
    # softmax normalization with temperature
        T = 0.00001
        flatten_grad = torch.flatten(grad, start_dim=1, end_dim=- 1)
        softmax_grad = torch.nn.Softmax(dim=1)(flatten_grad/T)
        unflatten_grad = torch.nn.Unflatten(1,(1,32,32))(softmax_grad)
        grad = unflatten_grad*np.clip(alpha, a_min=1e-6, a_max=None)
    else:
        raise NotImplementedError('Invalid normalization')

    return grad

def return_center(padded_input,w):
    return padded_input[:,:,w:2*w,w:2*w]

def gradmix_v2(x, y, grad, alpha=1, normalization='standard', stride=10, debug=False, rand_pos=False):
    '''Returns mixed inputs, pairs of targets, and lambda'''

    batch_size, c, w, h = np.array(x.size())
    if debug:
        index = torch.tensor([1,4,3,0,2]).cuda()
    else:
        index = torch.randperm(batch_size).cuda()

    mixed_y = [y, y[index]]

    max_criteria = torch.zeros([batch_size]).cuda()
    grad_1, grad_2 = grad, grad[index, :]
    mixed_x = torch.zeros_like(x).cuda()
    _mixed_lam = torch.zeros([batch_size]).cuda()

    normalized_grad_1 = normalize_grad(grad_1, alpha, normalization)
    padded_normalized_grad_1 = pad_zeros(normalized_grad_1, w,w,w,w)
    padded_x_1 = pad_zeros(x, w,w,w,w)

    # rand_pos is 0: double forloop
    # rand_pos is 1: random sampling the same number of time as if we are doing double for loop
    # rand_pos > 1: random sampling, with number *= rand_pos
    if rand_pos:
        total_iteration = int((len(range(0,w,stride))**2)/rand_pos)

    for ii, _i in enumerate(range(0,w,stride)):
        for jj, _j in enumerate(range(0,w,stride)):
            if rand_pos:
                i = np.random.randint(low=0, high=w)
                j = np.random.randint(low=0, high=w)
            else:
                i, j = _i, _j

            normalized_grad_2 = normalize_grad(grad_2, 1-alpha, normalization)
            padded_normalized_grad_2 = pad_zeros(normalized_grad_2, w-j,w+j,0+i,2*w-i)
            padded_x_2 = pad_zeros(x[index,:], w-j,w+j,0+i,2*w-i)

            M = padded_normalized_grad_1 / (padded_normalized_grad_1+padded_normalized_grad_2+1e-6)

            current_saliency = return_center(padded_normalized_grad_1 * M+(padded_normalized_grad_2 * (1-M)), w)
            criteria = current_saliency.sum(dim=[1,2,3])
            update_needed = ((criteria - max_criteria)>0)

            if update_needed.sum() >0:
                lambbda = return_center(M,w).mean(dim=[1,2,3]).detach().clone()
                M_adjusted = M.expand(-1,3,-1,-1)
                mixed_x[update_needed,:,:,:] = (return_center(padded_x_1 * M_adjusted + (padded_x_2 * (1-M_adjusted)), w)[update_needed]).detach().clone()

                _mixed_lam[update_needed] = lambbda[update_needed]
                max_criteria[update_needed] = criteria[update_needed].detach().clone()
            if rand_pos and (ii*len(range(0,w,stride))+jj+1)>=total_iteration:
                break
        if rand_pos and (ii*len(range(0,w,stride))+jj+1)>=total_iteration:
            break

#     grad_2, grad_1 = grad, grad[index, :]

#     normalized_grad_1 = normalize_grad(grad_1, alpha)
#     padded_normalized_grad_1 = pad_zeros(normalized_grad_1, w,w,w,w)
#     padded_x_1 = pad_zeros(x[index,:], w,w,w,w)

#     for i in range(0,w,stride):
#         for j in range(0,w,stride):
#             normalized_grad_2 = normalize_grad(grad_2, 1-alpha)
#             padded_normalized_grad_2 = pad_zeros(normalized_grad_2, w-j,w+j,0+i,2*w-i)
#             padded_x_2 = pad_zeros(x, w-j,w+j,0+i,2*w-i)

#             M = padded_normalized_grad_1 / (padded_normalized_grad_1+padded_normalized_grad_2+1e-6)

#             current_saliency = return_center(padded_normalized_grad_1 * M+(padded_normalized_grad_2 * (1-M)), w)
#             criteria = current_saliency.sum(dim=[1,2,3])
#             update_needed = ((criteria - max_criteria)>0)

#             if update_needed.sum() >0:
#                 lambbda = return_center(M,w).mean(dim=[1,2,3])
#                 M_adjusted = M.expand(-1,3,-1,-1)
#                 mixed_x[update_needed,:,:,:] = return_center(padded_x_1 * M_adjusted + (padded_x_2 * (1-M_adjusted)), w)[update_needed]
#                 _mixed_lam[update_needed] = lambbda[update_needed]
#                 max_criteria[update_needed] = criteria[update_needed]

    mixed_lam = [_mixed_lam.detach(), 1- _mixed_lam.detach()]

    del max_criteria, _mixed_lam, index, M, M_adjusted
    del padded_normalized_grad_1, padded_normalized_grad_2, normalized_grad_1, normalized_grad_2, padded_x_1, padded_x_2

    return mixed_x.detach(), mixed_y, mixed_lam

def pad_zeros(input_2b_padded, a, b, c, d):
    padded_input = torch.nn.functional.pad(input_2b_padded, [a, b, c, d], mode='constant', value=0.0)
    return padded_input

def reweighted_lam(mixed_y, mixed_lam, num_classes):
    y0 = to_one_hot(mixed_y[0], num_classes)
    y1 = to_one_hot(mixed_y[1], num_classes)

    return mixed_lam[0].unsqueeze(1)*y0 + mixed_lam[1].unsqueeze(1)*y1

def gradmix_v2_improved(x, y, grad, alpha=1, normalization='standard', stride=10, debug=False, rand_pos=False):
    '''Returns mixed inputs, pairs of targets, and lambda'''

    batch_size, c, w, h = np.array(x.size())
    if debug:
        index = torch.tensor([1,4,3,0,2]).cuda()
    else:
        index = torch.randperm(batch_size).cuda()

    mixed_y = [y, y[index]]

    max_criteria = torch.zeros([batch_size]).cuda()
    grad_1, grad_2 = grad.cuda(), grad[index, :].cuda()
    mixed_x = torch.zeros_like(x).cuda()
    _mixed_lam = torch.zeros([batch_size]).cuda()

    normalized_grad_1 = normalize_grad(grad_1, alpha, normalization)
    normalized_grad_2 = normalize_grad(grad_2, 1-alpha, normalization)
    padded_normalized_grad_1 = pad_zeros(normalized_grad_1, w,w,w,w)
    padded_x_1 = pad_zeros(x, w,w,w,w)

    # rand_pos is 0: double forloop
    # rand_pos is 1: random sampling the same number of time as if we are doing double for loop
    # rand_pos > 1: random sampling, with number *= rand_pos
    if rand_pos:
        total_iteration = int((w/stride)**2/rand_pos)
        coord = np.random.randint(low=0, high=w, size=(2,total_iteration))
#         rand_coord = np.unique(rand_coord, axis=1)
    else:
        _x = np.linspace(0, w, int(w/stride))
        _y = np.linspace(0, w, int(w/stride))
        _xv, _yv = np.meshgrid(_x, _y)
        coord = np.stack((_xv.astype(int).flatten(), _yv.astype(int).flatten()))
        coord = np.insert(coord, 0, np.array([0,w]),axis=1)
    
#     total_time_in_update = 0
#     update_counter = 0
    
    for _i in range(coord.shape[1]):
        i,j=coord[:,_i]
        padded_normalized_grad_2 = pad_zeros(normalized_grad_2, w-j,w+j,0+i,2*w-i)
        M = padded_normalized_grad_1 / (padded_normalized_grad_1+padded_normalized_grad_2+1e-6)

        current_saliency = return_center(padded_normalized_grad_1 * M+(padded_normalized_grad_2 * (1-M)), w)
        criteria = current_saliency.sum(dim=[1,2,3])
        update_needed = ((criteria - max_criteria)>0)

#         tic = time.perf_counter()
        if update_needed.sum() >0:
#             update_counter += 1
            lambbda = return_center(M,w).mean(dim=[1,2,3])
            padded_x_2 = pad_zeros(x[index,:], w-j,w+j,0+i,2*w-i)
            mixed_x[update_needed,:,:,:] = return_center(torch.mul(padded_x_1, M) + torch.mul(padded_x_2, 1-M), w)[update_needed]

            _mixed_lam[update_needed] = lambbda[update_needed]
            max_criteria[update_needed] = criteria[update_needed]
#         toc = time.perf_counter()
#         total_time_in_update += (toc-tic)

    mixed_lam = [_mixed_lam.detach(), 1- _mixed_lam.detach()]

    del max_criteria, _mixed_lam, index, M
    del padded_normalized_grad_1, padded_normalized_grad_2, normalized_grad_1, normalized_grad_2, padded_x_1, padded_x_2

    return mixed_x.detach(), mixed_y, mixed_lam
#     return mixed_x.detach(), mixed_y, mixed_lam, total_time_in_update, update_counter
