import glob
import os
import random
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from utils import *


def apply_gamma_high(x):
    result = x/255.0 ** random.uniform(0.75, 0.95) *255.0
    return np.uint8(np.clip(result, 0, 255))

def apply_gamma_low(x):
    return np.round(x/255.0 ** random.uniform(1.05, 1.3) *255.0).astype(np.uint8)

def ori(x):
    return x

def poisson_noise(x):
    x = np.float32(x)
    scale = random.uniform(0.1, 0.3)
    noise = np.random.poisson(x * scale) / scale
    return np.uint8(np.clip(x + noise, 0, 255))

def gaussian_noise(x):
    x = np.float32(x)
    mean = 0
    sigma = random.uniform(0.04, 0.2) # var ** 0.5
    noise = (np.random.normal(mean, sigma, x.shape)) * 255#.astype('uint8')
    return np.uint8(np.clip(x + noise, 0, 255))

def higher_contrast(image):
    alpha=random.uniform(1.2, 1.5)
    beta = np.mean(image) * (1 - alpha)
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def higher_contrast_large_ratio(image):
    alpha=random.uniform(2, 4)
    beta = np.mean(image) * (1 - alpha)
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def lower_contrast(image):
    alpha=random.uniform(0.5, 0.9)
    beta = np.mean(image) * (1 - alpha)
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def high_saturation(image, saturation_scale=random.uniform(1.3, 2)):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    s = cv2.multiply(s, saturation_scale)
    hsv_image = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

def low_saturation(image, saturation_scale=random.uniform(0.6, 0.9)):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    s = cv2.multiply(s, saturation_scale)
    hsv_image = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

class SICE_IR(Dataset):
    def __init__(self, img_dir, transform=None):
        self.base_dir = img_dir
        self.source_dir = img_dir +'/IR'

        self.source = [im_name for im_name in os.listdir(self.source_dir)
                       if im_name.split('.')[-1].lower() in ('jpg', 'png', 'bmp')]
        self.transform = transform

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        name = self.source[idx]
        img = cv2.imread(os.path.join(self.source_dir, name), cv2.IMREAD_GRAYSCALE)

        distortions = {1: gaussian_noise, 2: apply_gamma_low, 3: apply_gamma_high, 4: lower_contrast,
                       5: higher_contrast, 6: ori}

        img1_r = np.copy(img)
        img2_r = np.copy(img)

        noise_func = distortions.get(1)

        H = img.shape[0]
        W = img.shape[1]
        x = random.randint(0, H // 3 - 1)
        y = random.randint(0, W // 3 - 1)
        h = random.randint(H // 3 * 2, H - 1)
        w = random.randint(W // 3 * 2, W - 1)

        dist1 = random.randint(2, 6)
        dist_func1 = distortions.get(dist1)
        if random.randint(0, 5) < 2:
            img1 = noise_func(dist_func1(img))
        else:
            img1 = dist_func1(img)
        img1_r[x:h, y:w] = img1[x:h, y:w]

        x = random.randint(0, H // 3 - 1)
        y = random.randint(0, W // 3 - 1)
        h = random.randint(H // 3 * 2, H - 1)
        w = random.randint(W // 3 * 2, W - 1)

        dist2 = random.randint(2, 6)
        dist_func2 = distortions.get(dist2)
        if random.randint(0, 5) < 2:
            img2 = noise_func(dist_func2(img))
        else:
            img2 = dist_func2(img)
        img2_r[x:h, y:w] = img2[x:h, y:w]

        if self.transform:
            seed = torch.random.seed()
            torch.random.manual_seed(seed)
            img1_r = self.transform(img1_r)
            torch.random.manual_seed(seed)
            img2_r = self.transform(img2_r)

        sample = {'name': name, 'img1': img1_r, 'img2': img2_r}
        return sample


class SICE_VIS(Dataset):
    def __init__(self, img_dir, transform=None):
        self.base_dir = img_dir
        self.source_dir = img_dir +'/VIS_for_structure'

        self.source = [im_name for im_name in os.listdir(self.source_dir)
                       if im_name.split('.')[-1].lower() in ('jpg', 'png', 'bmp')]
        self.transform = transform

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        name = self.source[idx]
        img = cv2.imread(os.path.join(self.source_dir, name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        distortions = {1: gaussian_noise, 2: apply_gamma_low, 3: apply_gamma_high, 4: lower_contrast, 5: higher_contrast,
                       6: ori, 7: low_saturation}

        img1_r = np.copy(img)
        img2_r = np.copy(img)

        noise_func = distortions.get(1)

        H = img.shape[0]
        W = img.shape[1]
        x = random.randint(0, H // 3 - 1)
        y = random.randint(0, W // 3 - 1)
        h = random.randint(H // 3 * 2, H - 1)
        w = random.randint(W // 3 * 2, W - 1)

        dist1 = random.randint(2, 6)
        dist_func1 = distortions.get(dist1)
        if random.randint(0, 5) < 2:
            img1 = noise_func(dist_func1(img))
        else:
            img1 = dist_func1(img)
        img1_r[x:h, y:w] = img1[x:h, y:w]

        x = random.randint(0, H // 3 - 1)
        y = random.randint(0, W // 3 - 1)
        h = random.randint(H // 3 * 2, H - 1)
        w = random.randint(W // 3 * 2, W - 1)

        dist2 = random.randint(2, 6)
        dist_func2 = distortions.get(dist2)
        if random.randint(0, 5) < 2:
            img2 = noise_func(dist_func2(img))
        else:
            img2 = dist_func2(img)
        img2_r[x:h, y:w] = img2[x:h, y:w]

        if self.transform:
            seed = torch.random.seed()
            torch.random.manual_seed(seed)
            img1 = self.transform(img1)
            torch.random.manual_seed(seed)
            img2 = self.transform(img2)

        sample = {'name': name, 'img1': img1, 'img2': img2}
        return sample



class SICE_F_stru(Dataset):
    def __init__(self, img_dir, transform=None):
        self.base_dir = img_dir
        self.source1_dir = img_dir +'/VIS'
        self.source2_dir = img_dir + '/IR'
        print(self.source1_dir)
        print(self.source2_dir)

        self.source1 = [im_name for im_name in os.listdir(self.source1_dir)
                       if im_name.split('.')[-1].lower() in ('jpg', 'png', 'bmp')]
        self.source2 = [im_name for im_name in os.listdir(self.source2_dir)
                        if im_name.split('.')[-1].lower() in ('jpg', 'png', 'bmp')]
        self.transform = transform

    def __len__(self):
        return len(self.source1)

    def __getitem__(self, idx):
        name = self.source1[idx]
        img1 = cv2.imread(os.path.join(self.source1_dir, name))
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.imread(os.path.join(self.source2_dir, name), cv2.IMREAD_GRAYSCALE)

        height1, width1, _ = img1.shape
        if height1 < 280 or width1 < 280:
            img1 = cv2.resize(img1, (280, 280))
            img2 = cv2.resize(img2, (280, 280))

        distortions = {1: apply_gamma_low, 2: lower_contrast, 3: higher_contrast, 4: gaussian_noise, 5: ori,
                       6: low_saturation, 7: high_saturation}

        dist1 = random.randint(1, 7)
        dist_func1 = distortions.get(dist1)
        img1 = dist_func1(img1)

        dist2 = random.randint(1, 5)
        dist_func2 = distortions.get(dist2)
        img2 = dist_func2(img2)

        if self.transform:
            seed = torch.random.seed()
            torch.random.manual_seed(seed)
            img1 = self.transform(img1)
            torch.random.manual_seed(seed)
            img2 = self.transform(img2)

        sample = {'name': name, 'img1': img1, 'img2': img2}
        return sample


class SICE_F_attr(Dataset):
    def __init__(self, img_dir, transform=None):
        self.source_dir = img_dir +'/VIS'

        self.source = [im_name for im_name in os.listdir(self.source_dir)
                       if im_name.split('.')[-1].lower() in ('jpg', 'png', 'bmp')]
        self.transform = transform

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(self.source_dir, self.source[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        height, width, _ = img.shape
        if height <280 or width <280:
            img = cv2.resize(img, (280, 280))

        distortions = {1: apply_gamma_low, 2: higher_contrast, 3: high_saturation, 4: low_saturation, 5: ori, 6: lower_contrast}

        dist = random.randint(2, 6)
        dist_func = distortions.get(dist)
        img = dist_func(img)

        if self.transform:
            seed = torch.random.seed()
            torch.random.manual_seed(seed)
            img = self.transform(img)
        sample = {'img': img}
        return sample



class SICE_TEST(Dataset):
    def __init__(self, img_dir, transform=None):
        self.base_dir = img_dir
        self.source1_dir = img_dir +'/vis'
        print(self.source1_dir)
        self.source2_dir = img_dir +'/ir'
        print(self.source2_dir)

        self.source1 = [im_name for im_name in os.listdir(self.source1_dir)
                       if im_name.split('.')[-1].lower() in ('jpg', 'png', 'bmp')]
        self.source2 = [im_name for im_name in os.listdir(self.source2_dir)
                       if im_name.split('.')[-1].lower() in ('jpg', 'png', 'bmp')]
        self.transform = transform

    def __len__(self):
        return len(self.source1)

    def __getitem__(self, idx):
        name = self.source1[idx]
        img1 = cv2.imread(os.path.join(self.source1_dir, name))
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.imread(os.path.join(self.source2_dir, name), cv2.IMREAD_GRAYSCALE)

        height, width, _ = img1.shape
        if height % 4 != 0 or width % 4 != 0:
            img1 = cv2.resize(img1, (width // 4 * 4, height // 4 * 4))
            img2 = cv2.resize(img2, (width // 4 * 4, height // 4 * 4))

        if self.transform:
            seed = torch.random.seed()
            torch.random.manual_seed(seed)
            img1 = self.transform(img1)
            torch.random.manual_seed(seed)
            img2 = self.transform(img2)

        sample = {'name': name, 'img1': img1,'img2':img2}
        return sample


class SICE_A_eval(Dataset):
    def __init__(self, img_dir, transform=None):
        self.base_dir = img_dir
        self.source_dir = img_dir
        print(self.source_dir)

        self.source = [im_name for im_name in os.listdir(self.source_dir)
                       if im_name.split('.')[-1].lower() in ('jpg', 'png', 'bmp')]
        self.transform = transform

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        name = self.source[idx]
        img = cv2.imread(os.path.join(self.source_dir, name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        height, width, _ = img.shape
        if height % 4 != 0 or width % 4 != 0:
            img = cv2.resize(img, (width // 4 * 4, height // 4 * 4))

        if self.transform:
            seed = torch.random.seed()
            torch.random.manual_seed(seed)
            img = self.transform(img)

        sample = {'name': name, 'img': img}
        return sample
