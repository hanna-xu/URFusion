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

    # loss
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


def vector_contrastive_loss(vec_s, vec_p, vec_n, tau):
    for c in range(vec_s.size(0)):
        vec_s_tile_n = vec_s[c:c + 1, :].repeat(vec_n.size(1), 1)
        pos = - torch.norm(vec_s[c, :] - vec_p[c, :]) / tau
        neg = - torch.norm(vec_s_tile_n - vec_n[c,:, :], dim=1)/ tau
        softmax = - torch.logsumexp(pos, 0) + torch.logsumexp(neg, 0)
        if c == 0:
            softmaxs = softmax.unsqueeze(0)
        else:
            softmaxs = torch.cat((softmaxs, softmax.unsqueeze(0)), 0)
    return torch.mean(softmaxs, 0)


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


def box_filter(x, r):
    ''' Implements a box filter using PyTorch '''
    return F.avg_pool2d(x, (2 * r + 1, 2 * r + 1), stride=1, padding=r)


def guided_filter(X, G, r=2, eps=0.0003):
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


def fusion_A2V_contrastive_loss(f_A2V, s1_A2V, s2_A2V, normal_A2V, normal_A2V_1, normal_A2V_2, normal_A2V_3, tau=0.4):
    pos = - torch.norm(f_A2V - normal_A2V, dim=1) -torch.norm(f_A2V - normal_A2V_1, dim=1) \
          - torch.norm(f_A2V - normal_A2V_2, dim=1)-torch.norm(f_A2V - normal_A2V_3, dim=1)
    neg = - torch.norm(f_A2V - s1_A2V, dim=1) - torch.norm(f_A2V - s2_A2V, dim=1)
    for c in range(f_A2V.size(0)):
        softmax = - torch.logsumexp(pos[c]/ tau, 0) + torch.logsumexp(neg[c]/ tau, 0)
        if c == 0:
            softmaxs = softmax.unsqueeze(0)
        else:
            softmaxs = torch.cat((softmaxs, softmax.unsqueeze(0)), 0)
    return torch.mean(- pos)


def l0(x, e=0.4):
    y = torch.where(torch.abs(x) > torch.ones_like(x, dtype=torch.float) * e, torch.ones_like(x, dtype=torch.float),
                 torch.square(x) / (e * e))
    return torch.mean(y)


def normalization(x):
    b =x.shape[0]
    c=x.shape[1]
    h = x.shape[2]
    w = x.shape[3]
    max_val, _ = torch.max(x.reshape(b, c*h*w), dim=1)
    min_val, _ = torch.min(x.reshape(b, c*h*w), dim=1)
    max_val = max_val.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 3, h, w)
    min_val = min_val.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 3, h, w)
    x_nor = (x-min_val+1e-4)/(max_val-min_val+1e-4)
    x_norm_clip = torch.clamp(x_nor, 0, 1)
    return x_norm_clip


def channel_normalization(x, y, x_e, y_e):
    h = x.shape[2]
    w = x.shape[3]
    max_c_x, _ = torch.max(x[:, :, :, :], dim=2)
    max_c_x, _ = torch.max(max_c_x, dim=2)
    min_c_x, _ = torch.min(x[:, :, :, :], dim=2)
    min_c_x, _ = torch.min(min_c_x, dim=2)
    max_c_y, _ = torch.max(y[:, :, :, :], dim=2)
    max_c_y, _ = torch.max(max_c_y, dim=2)
    min_c_y, _ = torch.min(y[:, :, :, :], dim=2)
    min_c_y, _ = torch.min(min_c_y, dim=2)

    max_c, _ = torch.max(torch.stack((max_c_x, max_c_y), dim=1), dim=1)
    min_c, _ = torch.min(torch.stack((min_c_x, min_c_y), dim=1), dim=1)
    max_c = max_c.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, h, w)
    min_c = min_c.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, h, w)

    x_nor = torch.clamp((x - min_c) / (max_c - min_c + 1e-6), min=0, max=1)
    y_nor = torch.clamp((y - min_c) / (max_c - min_c + 1e-6), min=0, max=1)
    x_e_nor = torch.clamp((x_e - min_c) / (max_c - min_c + 1e-6), min=0, max=1)
    y_e_nor = torch.clamp((y_e - min_c) / (max_c - min_c + 1e-6), min=0, max=1)
    return x_nor, y_nor, x_e_nor, y_e_nor


def gradient_operator(x):
    kernel = torch.tensor([[0, 1.0, 0], [1.0, -2.0, 0], [0, 0, 0]], dtype=torch.float)
    kernel = kernel.view(1, 1, 3, 3).repeat(x.size(1), x.size(1), 1, 1)
    kernel = kernel.to(x.device)
    return nn.functional.conv2d(x, kernel, stride=1, padding=1)


def feature_gradient(x):
    for b in range(x.size(1)):
        grad = gradient_operator(x[:,b:b+1,:,:])
        if b==0:
            grads=grad
        else:
            grads=torch.cat([grads, grad], 1)
    return grads


def gaussian_kernel(size, sigma):
    """Generates a 2D Gaussian kernel."""
    kernel_1d = torch.tensor([np.exp(-(x - size // 2) ** 2 / (2 * sigma ** 2)) for x in range(size)]).float()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    kernel_2d /= kernel_2d.sum()
    return kernel_2d


def apply_gaussian_blur(feature_map, kernel_size=5, sigma=1.0, device=1):
    """Applies Gaussian blur to the input feature map."""
    channels, height, width = feature_map.shape[1], feature_map.shape[2], feature_map.shape[3]
    kernel = gaussian_kernel(kernel_size, sigma).unsqueeze(0).unsqueeze(0).to(device)  # Shape [1, 1, kernel_size, kernel_size]
    kernel = kernel.repeat(channels, 1, 1, 1)  # Shape [channels, 1, kernel_size, kernel_size]
    feature_map_blur = F.conv2d(feature_map, kernel, padding=kernel_size // 2, groups=channels, stride=1)
    return feature_map_blur


def shuffle(x):
    x_v = torch.reshape(x,(x.size(0),x.size(1),x.size(2)*x.size(3)))
    indices = torch.randperm(x_v.size(2))
    shuffled_x_v = x_v[:,:,indices]
    return shuffled_x_v


def cos_sim_rgb(x, y):
    cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    cos_sim_r = cos_sim(x[:, 0, :], y[:, 0, :])
    cos_sim_g = cos_sim(x[:, 1, :], y[:, 1, :])
    cos_sim_b = cos_sim(x[:, 2, :], y[:, 2, :])
    return cos_sim_r + cos_sim_g +cos_sim_b

def find_most_sim_mse(x):
    for c in range(x.size(0)):
        a = x[c:c+1, :, :, :].repeat(x.size(0), 1, 1, 1)
        mse_sim_results = torch.mean((a-x) **2, dim=[1, 2, 3])
        value, indice = mse_sim_results.topk(3, dim=0, largest=False)
        if c==0:
            indices=indice[1:]
        else:
            indices = torch.cat((indices, indice[1:]), dim=0)
    return indices


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


def find_most_sim(x):
    intensity = torch.mean(x, dim=[1, 2, 3]).unsqueeze(-1)
    color_abundant=color_abun(x)
    x_ycbcr = rgb2ycbcr(x)
    x_y = x_ycbcr[:,0,:,:]
    x_normalized = (x_y - x_y.mean()) / x_y.std()
    contrast = torch.mean(x_normalized ** 2, dim=[1,2])
    z1, _=torch.min(x, dim=1)
    z2, _ = torch.max(x, dim=1)
    saturation = torch.mean(z1/(z2+1e-6), dim=[1, 2])
    x_color_v = torch.cat((intensity, color_abundant, contrast.unsqueeze(-1)/1.5, saturation.unsqueeze(-1)), dim=1)
    for c in range(x.size(0)):
        a = x_color_v[c:c+1, :].repeat(x.size(0), 1)
        color_sim_results = torch.norm(a - x_color_v, dim=1)# cos_sim(a, x_color_v)# torch.mean((a-x_color_v) **2, dim=[1])
        value, indice = color_sim_results.topk(4, dim=0, largest=False)
        if c==0:
            indices=indice[1:].unsqueeze(0)
        else:
            indices = torch.cat((indices, indice[1:].unsqueeze(0)), dim=0)
    return indices


def unsim_patches(x):
    for c in range(x.size(0)):
        a = x[c:c+1, 0, :, :, :].repeat(x.size(0)*x.size(1), 1, 1, 1)
        x_reshape = x.view(x.size(0)*x.size(1), x.size(2), x.size(3), x.size(4))
        sim_results = torch.mean((a - x_reshape)**2, dim=[1, 2, 3])
        value, indice = sim_results.topk(x.size(0)*x.size(1)//3, dim=0, largest=True)
        unsim_patch = x_reshape[indice, :,:,:]
        if c==0:
            result = unsim_patch.unsqueeze(0)
        else:
            result = torch.cat((result, unsim_patch.unsqueeze(0)), dim=0)
    return result


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
    avg_intensity = F.avg_pool2d(enhances, rsize).mean(1)
    exp_loss = (avg_intensity - E).abs().mean()
    return exp_loss


def color_constency_loss(enhances):
    plane_avg = enhances.mean((2, 3))
    col_loss = torch.mean((plane_avg[:, 0] - plane_avg[:, 1]) ** 2
                          + (plane_avg[:, 1] - plane_avg[:, 2]) ** 2
                          + (plane_avg[:, 2] - plane_avg[:, 0]) ** 2)
    return col_loss


def color_constency_loss2(enhances, originals):
    enh_cols = enhances.mean((2, 3))
    ori_cols = originals.mean((2, 3))
    rg_ratio = (enh_cols[:, 0] / enh_cols[:, 1] - ori_cols[:, 0] / ori_cols[:, 1]).abs()
    gb_ratio = (enh_cols[:, 1] / enh_cols[:, 2] - ori_cols[:, 1] / ori_cols[:, 2]).abs()
    br_ratio = (enh_cols[:, 2] / enh_cols[:, 0] - ori_cols[:, 2] / ori_cols[:, 0]).abs()
    col_loss = (rg_ratio + gb_ratio + br_ratio).mean()
    return col_loss


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
    K1 = torch.tensor([0.3, 0.59, 0.1], dtype=torch.float32).view(1, 3, 1, 1).to(device)
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
    enh_avg = F.avg_pool2d(enh_gray, rsize)
    ori_avg = F.avg_pool2d(ori_gray, rsize)
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

# training helper functions
class Logger:
    TRAIN_INFO = '[TRAIN] - EPOCH {:d}/{:d}, Iters {:d}/{:d}, {:.1f} s/iter, \
LOSS / LOSS(AVG): {:.4f}/{:.4f}, Loss[spa,exp,col,tvA] / Loss(avg)  : {} / {}'.strip()

    VAL_INFO = '[Validation] - EPOCH {:d}/{:d} - Validation Avg. LOSS: {:.4f}, in {:.2f} secs  '
    VAL_INFO += '- ' + datetime.now().strftime('%X') + ' -'

    def __init__(self, n):
        self.val = np.zeros(n)
        self.sum = np.zeros(n)
        self.count = 0
        self.avg = np.zeros(n)

        self.val_losses = []

    def update(self, losses):
        self.val = np.array(losses)
        self.sum += self.val
        self.count += 1
        self.avg = self.sum / self.count

def gamma_tensor(img, gamma=0.85):
    return torch.pow(img, gamma)

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


# system path
def create_dir(path):
    'create directory if not exist'
    if isinstance(path, str):
        path = Path(path).expanduser().resolve()

    if path.exists():
        if path.is_dir():
            print('Output dir already exists.')
        else:
            sys.exit('[ERROR] You specified a file, not a folder. Please revise --outputDir')
    else:
        path.mkdir(parents=True)
    return path
