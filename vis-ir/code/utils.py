import os
import shutil
import sys

from datetime import datetime
from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import torch.nn.functional as F
import pytorch_msssim

def alpha_total_variation(A):
    '''
    Links: https://remi.flamary.com/demos/proxtv.html
           https://kornia.readthedocs.io/en/latest/_modules/kornia/losses/total_variation.html#total_variation
    '''
    delta_h = A[:, :, 1:, :] - A[:, :, :-1, :]
    delta_w = A[:, :, :, 1:] - A[:, :, :, :-1]

    tv = delta_h.abs().mean((2, 3)) + delta_w.abs().mean((2, 3))
    loss = torch.mean(tv.sum(1) / (A.shape[1] / 3))
    return loss


def contrastive_loss(all_samples, N, tau):
    for si in range(2 * N):
        this_sample = all_samples[si:si + 1, :, :, :]
        this_sample_tile = this_sample.repeat((2 * N, 1, 1, 1))
        sim = - torch.mean((this_sample_tile - all_samples) ** 2, dim=[1, 2, 3]) / tau
        sim = sim.unsqueeze(-1)
        if si == 0:
            similarities = sim
        else:
            similarities = torch.cat((similarities, sim), -1)

    for si in range(2 * N):
        sj = (si + N) % (2 * N)
        pos = similarities[si, sj]
        neg_num = 0
        for sk in range(2 * N):
            if sk != sj:
                neg = similarities[si, sk].unsqueeze(0)
                if neg_num == 0:
                    neg_num = neg_num + 1
                    negs = neg
                else:
                    negs = torch.cat((negs, neg), 0)
        softmax = - torch.logsumexp(pos, 0) + torch.logsumexp(negs, 0)
        softmax = softmax.unsqueeze(-1)

        if si == 0:
            softmaxes = softmax
        else:
            softmaxes = torch.cat((softmaxes, softmax), 0)

    contrast_loss = torch.mean(softmaxes, 0)
    return contrast_loss

def box_filter(x, r):
    ''' Implements a box filter using PyTorch '''
    return F.avg_pool2d(x, (2 * r + 1, 2 * r + 1), stride=1, padding=r)


def guided_filter(X, G, r=1, eps=0.0001):
    mean_G = box_filter(G, r)
    mean_X = box_filter(X, r)
    mean_Ip = box_filter(G * X, r)
    cov_Ip = mean_Ip - mean_G * mean_X

    mean_GG = box_filter(G * G, r)
    var_G = mean_GG - mean_G * mean_G

    a = cov_Ip / (var_G + eps)
    b = mean_X - a * mean_G

    mean_a = box_filter(a, r)
    mean_b = box_filter(b, r)

    q = mean_a * G + mean_b
    return q


def gradient_operator(x):
    kernel_x = torch.tensor([[0, 0, 0], [1.0, -1.0, 0], [0, 0, 0]], dtype=torch.float)
    kernel_y = torch.tensor([[0, 1.0, 0], [0, -1.0, 0], [0, 0, 0]], dtype=torch.float)
    kernel_x = kernel_x.view(1, 1, 3, 3).repeat(x.size(1), x.size(1), 1, 1)
    kernel_y = kernel_y.view(1, 1, 3, 3).repeat(x.size(1), x.size(1), 1, 1)
    kernel_x = kernel_x.to(x.device)
    kernel_y = kernel_y.to(x.device)
    gradient_x = torch.abs(nn.functional.conv2d(x, kernel_x, stride=1, padding=1))
    gradient_y = torch.abs(nn.functional.conv2d(x, kernel_y, stride=1, padding=1))
    return gradient_x + gradient_y


def ssim_loss(x, y):
    loss_ssim_R = 1 - pytorch_msssim.ssim(x[:, 0:1, :, :], y[:, 0:1, :, :], data_range=1, size_average=True)
    loss_ssim_G = 1 - pytorch_msssim.ssim(x[:, 1:2, :, :], y[:, 1:2, :, :], data_range=1, size_average=True)
    loss_ssim_B = 1 - pytorch_msssim.ssim(x[:, 2:3, :, :], y[:, 2:3, :, :], data_range=1, size_average=True)
    loss_ssim = (loss_ssim_R + loss_ssim_G + loss_ssim_B).mean()
    return loss_ssim

def weighted_sim_loss(x, y, ref_x, ref_y):
    channel_max_x, _ = torch.max(torch.abs(ref_x).view(x.size(0), x.size(1), x.size(2)*x.size(3)), dim=2)
    channel_max_y, _ = torch.max(torch.abs(ref_y).view(x.size(0), x.size(1), x.size(2)*x.size(3)), dim=2)
    channel_max_xy = torch.stack([channel_max_x, channel_max_y], dim=2)
    channel_max, _ = torch.max(channel_max_xy, dim=2)
    channel_max = channel_max.unsqueeze(2).unsqueeze(2)
    channel_max = channel_max.repeat(1, 1, x.size(2), x.size(3)) +1e-4
    return torch.mean((x/channel_max-y/channel_max)**2)


def l0(x, e=0.4):
    y = torch.where(torch.abs(x) > torch.ones_like(x, dtype=torch.float) * e, torch.ones_like(x, dtype=torch.float),
                 torch.square(x) / (e * e))
    return torch.mean(y)

def normalization(x):
    b =x.shape[0]
    c=x.shape[1]
    h = x.shape[2]
    w = x.shape[3]
    max_x, _ = torch.max(x.reshape(b, c, h*w), dim=2)
    min_x, _ = torch.min(x.reshape(b, c, h*w), dim=2)
    max_x = max_x.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, h, w)
    min_x = min_x.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, h, w)
    a = 2 / (max_x - min_x + 1e-6)
    b = (-max_x - min_x) / (max_x - min_x + 1e-6)
    return a * x + b

def laplacian_operator(x):
    kernel = torch.tensor([[0, 1.0, 0], [1.0, -4.0, 1.0], [0, 1.0, 0]], dtype=torch.float)
    kernel = kernel.view(1, 1, 3, 3).repeat(x.size(1), x.size(1), 1, 1)
    kernel = kernel.to(x.device)
    return nn.functional.conv2d(x, kernel, stride=1, padding=1)

def shuffle(x):
    x_v = torch.reshape(x,(x.size(0),x.size(1),x.size(2)*x.size(3)))
    indices = torch.randperm(x_v.size(2))
    shuffled_x_v = x_v[:,:,indices]
    return shuffled_x_v

def color_abun(x):
    R = x[:,0,:,:]
    G = x[:,1,:,:]
    B = x[:,2,:,:]
    rg = torch.abs(R - G)
    yb = torch.abs(0.5 * (R + G) - B)

    rbMean = torch.mean(rg, dim=[1, 2])
    rbStd = torch.std(rg, dim=[1, 2])
    ybMean = torch.mean(yb, dim=[1, 2])
    ybStd = torch.std(yb, dim=[1, 2])

    stdRoot = torch.sqrt((rbStd ** 2) + (ybStd ** 2))
    meanRoot = torch.sqrt((rbMean ** 2) + (ybMean ** 2))

    c=stdRoot + (0.3 * meanRoot)
    return c.unsqueeze(-1)

def rgb2gray(x):
    return 0.03 * x[:, 0:1, :, :] + 0.59 * x[:, 1:2, :, :] + 0.11 * x[:, 2:3, :, :]

def saturation(img):
	batchsize = img.size(0)
	dim = img.size(1)
	rgb_max, _ = torch.max(img, dim=1)
	rgb_min, _ = torch.min(img, dim=1)
	saturation = torch.mean((rgb_max - rgb_min) / (rgb_max), dim=[1, 2])
	return saturation

def apply_gamma(rgb):
    T = 0.0031308
    T_tensor = np.ones_like(rgb) * T
    combine = np.concatenate([np.expand_dims(T_tensor, axis=-1), np.expand_dims(rgb, axis=-1)], axis=-1)
    rgb1 = np.max(combine, axis=-1)
    return np.where(rgb < T, 12.92 * rgb, (1.055 * np.power(np.abs(rgb1), 1 / 2.4) - 0.055))  # 2.4

def rgb2gray(tensor):
    R = tensor[:, 0:1, :, :]
    G = tensor[:, 0:1, :, :]
    B = tensor[:, 0:1, :, :]
    return 0.2989 * R + 0.5870 * G + 0.1140 * B

def exposure_control_loss(enhances, rsize=16, E=0.6):
    avg_intensity = F.avg_pool2d(enhances, rsize).mean(1)  # to gray: (R+G+B)/3
    exp_loss = (avg_intensity - E).abs().mean()
    return exp_loss

# Color constancy loss via gray-world assumption.   In use.
def color_constency_loss(enhances):
    plane_avg = enhances.mean((2, 3))
    col_loss = torch.mean((plane_avg[:, 0] - plane_avg[:, 1]) ** 2
                          + (plane_avg[:, 1] - plane_avg[:, 2]) ** 2
                          + (plane_avg[:, 2] - plane_avg[:, 0]) ** 2)
    return col_loss

# Averaged color component ratio preserving loss.  Not in use.
def color_constency_loss2(enhances, originals):
    enh_cols = enhances.mean((2, 3))
    ori_cols = originals.mean((2, 3))
    rg_ratio = (enh_cols[:, 0] / enh_cols[:, 1] - ori_cols[:, 0] / ori_cols[:, 1]).abs()
    gb_ratio = (enh_cols[:, 1] / enh_cols[:, 2] - ori_cols[:, 1] / ori_cols[:, 2]).abs()
    br_ratio = (enh_cols[:, 2] / enh_cols[:, 0] - ori_cols[:, 2] / ori_cols[:, 0]).abs()
    col_loss = (rg_ratio + gb_ratio + br_ratio).mean()
    return col_loss

# pixel-wise color component ratio preserving loss. Not in use.
def anti_color_shift_loss(enhances, originals):
    def solver(c1, c2, d1, d2):
        pos = (c1 > 0) & (c2 > 0) & (d1 > 0) & (d2 > 0)
        return torch.mean((c1[pos] / c2[pos] - d1[pos] / d2[pos]) ** 2)

    enh_avg = F.avg_pool2d(enhances, 4)
    ori_avg = F.avg_pool2d(originals, 4)

    rg_loss = solver(enh_avg[:, 0, ...], enh_avg[:, 1, ...],
                     ori_avg[:, 0, ...], ori_avg[:, 1, ...])
    gb_loss = solver(enh_avg[:, 1, ...], enh_avg[:, 2, ...],
                     ori_avg[:, 1, ...], ori_avg[:, 2, ...])
    br_loss = solver(enh_avg[:, 2, ...], enh_avg[:, 0, ...],
                     ori_avg[:, 2, ...], ori_avg[:, 0, ...])

    anti_shift_loss = rg_loss + gb_loss + br_loss
    if torch.any(torch.isnan(anti_shift_loss)).item():
        sys.exit('Color Constancy loss is nan')
    return anti_shift_loss


def get_kernels(device):
    # weighted RGB to gray
    K1 = torch.tensor([0.3, 0.59, 0.1], dtype=torch.float32).view(1, 3, 1, 1).to(device)
    # K1 = torch.tensor([1 / 3, 1 / 3, 1 / 3], dtype=torch.float32).view(1, 3, 1, 1).to(device)

    # kernel for neighbor diff
    K2 = torch.tensor([[[0, -1, 0], [0, 1, 0], [0, 0, 0]],
                       [[0, 0, 0], [0, 1, 0], [0, -1, 0]],
                       [[0, 0, 0], [-1, 1, 0], [0, 0, 0]],
                       [[0, 0, 0], [0, 1, -1], [0, 0, 0]]], dtype=torch.float32)
    K2 = K2.unsqueeze(1).to(device)
    return K1, K2


def spatial_consistency_loss(enhances, originals, to_gray, neigh_diff, rsize=4):
    # convert to gray
    enh_gray = F.conv2d(enhances, to_gray)
    ori_gray = F.conv2d(originals, to_gray)

    # average intensity of local regision
    enh_avg = F.avg_pool2d(enh_gray, rsize)
    ori_avg = F.avg_pool2d(ori_gray, rsize)

    # calculate spatial consistency loss via convolution
    enh_pad = F.pad(enh_avg, (1, 1, 1, 1), mode='replicate')
    ori_pad = F.pad(ori_avg, (1, 1, 1, 1), mode='replicate')
    enh_diff = F.conv2d(enh_pad, neigh_diff)
    ori_diff = F.conv2d(ori_pad, neigh_diff)

    spa_loss = torch.pow((enh_diff - ori_diff), 2).sum(1).mean()
    return spa_loss


def rgb2ycbcr(img_rgb):
    img_rgb=torch.clamp(img_rgb, min=0,max=1)
    R = img_rgb[:, 0, :, :] * 255
    G = img_rgb[:, 1, :, :] * 255
    B = img_rgb[:, 2, :, :] * 255

    Y = 0.257 * R + 0.504 * G + 0.098 * B + 16.0
    Cb = -0.148 * R - 0.291 * G + 0.439 * B + 128.0
    Cr = 0.439 * R - 0.368 * G - 0.071 * B + 128.0
    img_ycbcr = torch.stack((Y, Cb, Cr), dim=1)
    return torch.clamp(img_ycbcr/255.0, min=0,max=1)

def ycbcr2rgb(img_ycbcr):
    img_ycbcr = torch.clamp(img_ycbcr, min=0, max=1)
    Y = img_ycbcr[:, 0, :, :] * 255
    Cb = img_ycbcr[:, 1, :, :] * 255
    Cr = img_ycbcr[:, 2, :, :] * 255
    R=1.164 * (Y - 16) + 1.596 * (Cr - 128)
    G = 1.164 * (Y - 16) - 0.813 * (Cr - 128) - 0.392 * (Cb - 128)
    B = 1.164 * (Y - 16) + 2.017 * (Cb - 128)
    img_rgb = torch.stack((R, G, B), dim=1)
    return torch.clamp(img_rgb/255.0, min=0,max=1)


def rgb2hue(img_rgb):
    img_rgb = torch.clamp(img_rgb, min=0, max=1)
    R = img_rgb[:, 0:1, :, :]
    G = img_rgb[:, 1:2, :, :]
    B = img_rgb[:, 2:3, :, :]
    RGB_max, _ = torch.max(img_rgb, dim=1, keepdim=True)
    RGB_min, _ = torch.min(img_rgb, dim=1, keepdim=True)
    mask1 = (RGB_max == R).float()
    mask2 = (RGB_max == G).float() * (1 - (mask1 == 1).float())
    mask3 = (RGB_max == B).float() * (1 - (mask1 == 1).float()) * (1 - (mask2 == 1).float())
    hue1 = 60 * (G - B) / (R - RGB_min + 1e-6)
    hue2 = 120 + 60 * (B - R) / (G - RGB_min + 1e-6)
    hue3 = 240 + 60 * (R - G) / (B - RGB_min + 1e-6)
    H = mask1 * hue1 + mask2 * hue2 + mask3 * hue3
    H =  torch.where(H <0, H+360, H)
    H_max, _ = torch.max(H.reshape(H.size(0)*H.size(2)*H.size(3), 1), dim=0)
    H_min, _ = torch.min(H.reshape(H.size(0)*H.size(2)*H.size(3), 1), dim=0)
    return torch.clamp(H/360.0, min=0, max=1)

def color_channel_fuse(x, y, tau):
    x = x * 255.0
    y = y * 255.0
    mask = ((x == tau) & (y == tau)).float()
    fenmu = mask * torch.ones_like(x) + (1 - mask) * (torch.abs(x - tau) + torch.abs(y - tau))
    fuse = (x * torch.abs(x - tau) + y * torch.abs(y - tau)) / fenmu
    result = fuse * (1 - mask) + torch.ones_like(x) * 128.0 * mask
    return torch.clamp(result/255.0,min=0, max=1)

def color_hue_fuse(x, y, tau=0.5):
    return (x+y)/2

def save_ckpt(state, is_best, experiment, epoch, ckpt_dir):
    filename = os.path.join(ckpt_dir, f'{experiment}_ckpt.pth')
    torch.save(state, filename)
    if is_best:
        print(f'[BEST MODEL] Saving best model, obtained on epoch = {epoch + 1}')
        shutil.copy(filename, os.path.join(ckpt_dir, f'{experiment}_best_model.pth'))

def gamma_correction(img, gamma):
    return np.power(img, gamma)


def gamma_like(img, enhanced):
    x, y = img.mean(), enhanced.mean()
    gamma = np.log(y) / np.log(x)
    return gamma_correction(img, gamma)


def to_numpy(t, squeeze=False, to_HWC=True):
    x = t.detach().cpu().numpy()
    if squeeze:
        x = x.squeeze()
    if to_HWC:
        x = x.transpose((1, 2, 0))
    return x

def putText(im, *args):
    text, pos, font, size, color, scale = args
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    im = cv2.putText(im, text, pos, font, size, color, scale)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im


def row_arrange(inp, fixed, adaptive, algo):
    if algo.shape != fixed.shape:
        algo = cv2.resize(algo, (fixed.shape[1], fixed.shape[0]))
    pos = (25, 50)
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (128 / 255., 117 / 255., 0.)

    inp = putText(inp, 'Input', pos, font, 2, color, 3)
    fixed = putText(fixed, 'Gamma(fixed=0.4)', pos, font, 2, color, 3)
    adaptive = putText(adaptive, 'Gamma(adaptive)', pos, font, 2, color, 3)
    algo = putText(algo, 'ZeroDCE', pos, font, 2, color, 3)
    return cv2.hconcat([inp, fixed, adaptive, algo])

def make_grid(dataset, vsep=8):
    n = len(dataset)
    img = to_numpy(dataset[0]['img'])
    h, w, _ = img.shape
    grid = np.ones((n * h + vsep * (n - 1), 4 * w, 3), dtype=np.float32)
    return grid, vsep
