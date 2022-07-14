import torch
import torchvision
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.optim as optim

from tqdm import trange
import wandb

import numpy as np
import os
import random
import ipdb
import time

from mixup import to_one_hot

import torch.autograd.profiler as profiler

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
    # rand_pos > 1: random sampling, with number /= rand_pos
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

    mixed_lam = [_mixed_lam.detach(), 1- _mixed_lam.detach()]

    del max_criteria, _mixed_lam, index, M, M_adjusted
    del padded_normalized_grad_1, padded_normalized_grad_2, normalized_grad_1, normalized_grad_2, padded_x_1, padded_x_2

    return mixed_x.detach(), mixed_y, mixed_lam

def pad_zeros(input_2b_padded, left, right, top, bottom):
    padded_input = torch.nn.functional.pad(input_2b_padded, [left, right, top, bottom], mode='constant', value=0.0)
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
    # rand_pos > 1: random sampling, with number /= rand_pos
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
    
    for _i in range(coord.shape[1]):
        i,j=coord[:,_i]
        padded_normalized_grad_2 = pad_zeros(normalized_grad_2, w-j,w+j,0+i,2*w-i)
        M = padded_normalized_grad_1 / (padded_normalized_grad_1+padded_normalized_grad_2+1e-6)

        current_saliency = return_center(padded_normalized_grad_1 * M+(padded_normalized_grad_2 * (1-M)), w)
        criteria = current_saliency.sum(dim=[1,2,3])
        update_needed = ((criteria - max_criteria)>0)

        if update_needed.sum() >0:
            lambbda = return_center(M,w).mean(dim=[1,2,3])
            padded_x_2 = pad_zeros(x[index,:], w-j,w+j,0+i,2*w-i)
            mixed_x[update_needed,:,:,:] = return_center(torch.mul(padded_x_1, M) + torch.mul(padded_x_2, 1-M), w)[update_needed]

            _mixed_lam[update_needed] = lambbda[update_needed]
            max_criteria[update_needed] = criteria[update_needed]

    mixed_lam = [_mixed_lam.detach(), 1- _mixed_lam.detach()]

    del max_criteria, _mixed_lam, index, M
    del padded_normalized_grad_1, padded_normalized_grad_2, normalized_grad_1, normalized_grad_2, padded_x_1, padded_x_2

    return mixed_x.detach(), mixed_y, mixed_lam

def gradmix_v2_improved_v2(x, y, grad, alpha=1, normalization='standard', debug=False, rand_pos=1):
    '''Returns mixed inputs, pairs of targets, and lambda'''

    batch_size, c, w, h = np.array(x.size())
    if debug:
        index = torch.tensor([1,4,3,0,2]).cuda()
    else:
#         index = torch.range(start=99, end=0, step=-1, dtype=int).cuda()
        index = torch.randperm(batch_size).cuda()

    mixed_y = [y, y[index]]

    max_criteria = torch.zeros([batch_size]).cuda()
    best_ij = torch.empty([batch_size, 2], dtype=int).cuda()
    best_ij[:,0]=32
    best_ij[:,1]=32
    grad_1, grad_2 = grad.cuda(), grad[index, :].cuda()
    mixed_x = torch.zeros_like(x).cuda()
    _mixed_lam = torch.zeros([batch_size]).cuda()

    normalized_grad_1 = normalize_grad(grad_1, alpha, normalization)
    normalized_grad_2 = normalize_grad(grad_2, 1-alpha, normalization)
    padded_normalized_grad_1 = pad_zeros(normalized_grad_1, w,w,w,w)
    padded_x_1 = pad_zeros(x, w,w,w,w)

    # rand_pos is 0: double forloop
    # rand_pos is 1: random sampling the same number of time as if we are doing double for loop
    # rand_pos > 1: random sampling, with number /= rand_pos
#     if rand_pos:
#         total_iteration = int((2*(w-1))**2)
#         coord = np.random.randint(low=1, high=2*w-1, size=(2,total_iteration))
#         ipdb.set_trace()
#         print(coord.shape)
#         rand_coord = np.unique(rand_coord, axis=1)
#     else:
    stride=2
    rand_pos=1
    total_iteration = int((w/stride)**2/rand_pos)
    coord = np.random.randint(low=0, high=w, size=(2,total_iteration))
    # possible_positions = int((2*(w-1)))
    # _x = np.linspace(1, 2*w-1, possible_positions)
    # _xv, _yv = np.meshgrid(_x, _x)
    # coord = np.stack((_xv.astype(int).flatten(), _yv.astype(int).flatten()))
    # coord = coord[:,np.random.permutation(coord.shape[1])][:,:int(possible_positions**2*rand_pos)]
    # coord[:,0]=np.array([w,w])
    
    for _i in range(coord.shape[1]):
        i,j=coord[:,_i]
        padded_normalized_grad_2 = pad_zeros(normalized_grad_2, left=j,right=2*w-j,top=i,bottom=2*w-i)
        M = padded_normalized_grad_1 / (padded_normalized_grad_1+padded_normalized_grad_2+1e-6)

        current_saliency = return_center(padded_normalized_grad_1 * M+(padded_normalized_grad_2 * (1-M)), w)
        criteria = current_saliency.sum(dim=[1,2,3])
        update_needed = ((criteria - max_criteria)>0)
        
        if update_needed.sum() >0:
            best_ij[update_needed,:] = torch.tensor([i,j]).cuda()
            max_criteria[update_needed] = criteria[update_needed]

    for img in range(batch_size):
        i,j = best_ij[img,0].item(), best_ij[img,1].item()
        padded_normalized_grad_2 = pad_zeros(normalized_grad_2[img], left=j,right=2*w-j,top=i,bottom=2*w-i).unsqueeze(0)
        M = (padded_normalized_grad_1[img] / (padded_normalized_grad_1[img]+padded_normalized_grad_2+1e-6))
        lambbda = return_center(M,w).mean(dim=[1,2,3])
        padded_x_2 = pad_zeros(x[index,:][img], left=j,right=2*w-j,top=i,bottom=2*w-i).unsqueeze(0)
        mixed_x[img,:,:,:] = return_center(torch.mul(padded_x_1[img].unsqueeze(0), M) + torch.mul(padded_x_2, 1-M), w)
        _mixed_lam[img] = lambbda

    mixed_lam = [_mixed_lam.detach(), 1- _mixed_lam.detach()]

    del max_criteria, _mixed_lam, index, M
    del padded_normalized_grad_1, padded_normalized_grad_2, normalized_grad_1, normalized_grad_2, padded_x_1, padded_x_2, best_ij

    return mixed_x.detach(), mixed_y, mixed_lam

def gradmix_v2_improved_v3(x, y, grad, alpha=1, normalization='standard', debug=False, rand_pos=1):
    '''Returns mixed inputs, pairs of targets, and lambda'''

    batch_size, c, w, h = np.array(x.size())
    if debug:
        index = torch.tensor([1,4,3,0,2]).cuda()
    else:
#         index = torch.range(start=99, end=0, step=-1, dtype=int, device = x.device)
        index = torch.randperm(batch_size, device = x.device)

    mixed_y = [y, y[index]]

    max_criteria = torch.zeros([batch_size], device = x.device)
    best_ij = torch.empty([batch_size, 2], dtype=int, device = x.device)
    best_ij[:,0]=32
    best_ij[:,1]=32
    grad_1, grad_2 = grad, grad[index, :]
    mixed_x = torch.zeros_like(x, device = x.device)
    _mixed_lam = torch.zeros([batch_size], device = x.device)

    normalized_grad_1 = normalize_grad(grad_1, alpha, normalization)
    normalized_grad_2 = normalize_grad(grad_2, 1-alpha, normalization)
    padded_normalized_grad_1 = pad_zeros(normalized_grad_1, w,w,w,w)
    padded_x_1 = pad_zeros(x, w,w,w,w)

    possible_positions = int((2*w-1))
    _x = np.linspace(1, 2*w-1, possible_positions, dtype=int)
    _xv, _yv = np.meshgrid(_x, _x)
    coord = np.stack((_xv.flatten(), _yv.flatten()))
    coord = coord[:,np.random.permutation(coord.shape[1])][:,:int(possible_positions**2*rand_pos)]
    coord[:,0]=np.array([w,w])

    for _i in range(coord.shape[1]):
        i,j=coord[:,_i]
        padded_normalized_grad_2 = pad_zeros(normalized_grad_2, left=j,right=2*w-j,top=i,bottom=2*w-i)
        M = padded_normalized_grad_1 / (padded_normalized_grad_1+padded_normalized_grad_2+1e-6)

        current_saliency = return_center(padded_normalized_grad_1 * M+(padded_normalized_grad_2 * (1-M)), w)
        criteria = current_saliency.sum(dim=[1,2,3])
        update_needed = ((criteria - max_criteria)>0)
        
        if update_needed.sum() >0:
            best_ij[update_needed,:] = torch.tensor([i,j], device = x.device)
            max_criteria[update_needed] = criteria[update_needed]
        
    padded_normalized_grad_2 = pad_zeros(normalized_grad_2, w, w, w, w)
    padded_x_2 = pad_zeros(x[index,:], w,w,w,w)
    
    theta = torch.eye(2,3,device=x.device).repeat(batch_size,1,1)
    theta[:,0,2] = 2*(w-best_ij[:,1])/(3*w)
    theta[:,1,2] = 2*(w-best_ij[:,0])/(3*w)
    size = torch.Size((batch_size,c,3*w,3*w))
    grid = F.affine_grid(theta, size, align_corners=False)

    translated_padded_normalized_grad_2 = F.grid_sample(padded_normalized_grad_2,
                                                         grid,
                                                         mode='nearest',
                                                         padding_mode ='zeros',
                                                         align_corners=False)
    M = (padded_normalized_grad_1 / (padded_normalized_grad_1+translated_padded_normalized_grad_2+1e-6))
    lambbda = return_center(M,w).mean(dim=[1,2,3])
    translated_padded_x_2 = F.grid_sample(padded_x_2,
                                          grid,
                                          mode='nearest',
                                          padding_mode ='zeros',
                                          align_corners=False)
    mixed_x = return_center(torch.mul(padded_x_1, M) + torch.mul(translated_padded_x_2, 1-M), w)
    _mixed_lam = lambbda

    mixed_lam = [_mixed_lam.detach(), 1- _mixed_lam.detach()]

    del max_criteria, _mixed_lam, index, M, grid, size, theta, translated_padded_x_2
    del padded_normalized_grad_1, padded_normalized_grad_2, normalized_grad_1, normalized_grad_2, padded_x_1, padded_x_2, best_ij

    return mixed_x.detach(), mixed_y, mixed_lam

def gradmix_v2_improved_v4(x, y, grad, alpha=1, normalization='standard', debug=False, rand_pos=1):
    '''Returns mixed inputs, pairs of targets, and lambda'''
####################################################################################
    # init a bunch of variables
    with profiler.record_function("init variables"):
        batch_size, c, w, h = np.array(x.size())
        if debug:
            index = torch.tensor([1,4,3,0,2]).cuda()
        else:
            # comment/uncomment here for reproducing the same results
#             index = torch.range(start=99, end=0, step=-1, dtype=int, device = x.device)
            index = torch.randperm(batch_size, device = x.device)

        mixed_y = [y, y[index]]

        max_criteria = torch.zeros([batch_size], device = x.device)
        best_ij = torch.empty([batch_size, 2], dtype=int, device = x.device)
        best_ij[:,0]=32
        best_ij[:,1]=32
        grad_1, grad_2 = grad, grad[index, :]
        mixed_x = torch.zeros_like(x, device = x.device)
        _mixed_lam = torch.zeros([batch_size], device = x.device)

        normalized_grad_1 = normalize_grad(grad_1, alpha, normalization)
        normalized_grad_2 = normalize_grad(grad_2, 1-alpha, normalization)
        padded_normalized_grad_1 = pad_zeros(normalized_grad_1, w,w,w,w)
        padded_normalized_grad_2 = pad_zeros(normalized_grad_2, w,w,w,w)
        padded_x_1 = pad_zeros(x, w,w,w,w)
        padded_x_2 = pad_zeros(x[index,:], w,w,w,w)

        possible_positions = int((2*w-1))
        _x = torch.linspace(1, 2*w-1, possible_positions, dtype=int, device=x.device)
        _xv, _yv = torch.meshgrid(_x, _x)
        coord = torch.stack((_xv.flatten(), _yv.flatten()))
        # comment/uncomment here for reproducing the same results
        coord = coord[:,np.random.permutation(coord.shape[1])][:,:int(possible_positions**2*rand_pos)]
#         coord = coord[:,:int(possible_positions**2*rand_pos)]
        coord[0,0],coord[1,0] = w,w
    
####################################################################################
    # iterate over images and find the best position that maximizes saliency
    with profiler.record_function("iterate over images"):
        theta = torch.eye(2,3,device=x.device).repeat(coord.shape[1],1,1)
        theta[:,0,2] = 2*(w-coord[1,:])/(3*w)
        theta[:,1,2] = 2*(w-coord[0,:])/(3*w)
        size = torch.Size((coord.shape[1],1,3*w,3*w))
        grid = F.affine_grid(theta, size, align_corners=False)
        
        for img in range(batch_size):
            single_padded_normalized_grad_2 = padded_normalized_grad_2[img].expand(coord.shape[1],1,3*w,3*w)
            translated_single_padded_normalized_grad_2 = F.grid_sample(single_padded_normalized_grad_2,
                                                                 grid,
                                                                 mode='nearest',
                                                                 padding_mode ='zeros',
                                                                 align_corners=False)

            single_padded_normalized_grad_1 = padded_normalized_grad_1[img].expand(coord.shape[1],1,3*w,3*w)

            M =  single_padded_normalized_grad_1 / (single_padded_normalized_grad_1+translated_single_padded_normalized_grad_2+1e-6)

            saliency = return_center(single_padded_normalized_grad_1 * M+(translated_single_padded_normalized_grad_2 * (1-M)), w)

            best_ij[img,:] = coord[:,saliency.sum(dim=[1,2,3]).argmax()]

####################################################################################
    # update mixed images
    with profiler.record_function("update mixed image"):
        theta = torch.eye(2,3,device=x.device).repeat(batch_size,1,1)
        theta[:,0,2] = 2*(w-best_ij[:,1])/(3*w)
        theta[:,1,2] = 2*(w-best_ij[:,0])/(3*w)
        size = torch.Size((batch_size,c,3*w,3*w))
        grid = F.affine_grid(theta, size, align_corners=False)

        translated_padded_normalized_grad_2 = F.grid_sample(padded_normalized_grad_2,
                                                             grid,
                                                             mode='nearest',
                                                             padding_mode ='zeros',
                                                             align_corners=False)
        M = (padded_normalized_grad_1 / (padded_normalized_grad_1+translated_padded_normalized_grad_2+1e-6))
        lambbda = return_center(M,w).mean(dim=[1,2,3])
        translated_padded_x_2 = F.grid_sample(padded_x_2,
                                              grid,
                                              mode='nearest',
                                              padding_mode ='zeros',
                                              align_corners=False)
        mixed_x = return_center(torch.mul(padded_x_1, M) + torch.mul(translated_padded_x_2, 1-M), w)
        _mixed_lam = lambbda

        mixed_lam = [_mixed_lam.detach(), 1- _mixed_lam.detach()]

        del max_criteria, _mixed_lam, index, M, grid, size, theta, translated_padded_x_2, coord
        del padded_normalized_grad_1, padded_normalized_grad_2, normalized_grad_1, normalized_grad_2, padded_x_1, padded_x_2, best_ij

    return mixed_x.detach(), mixed_y, mixed_lam
