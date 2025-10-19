import glob
import os
import random
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from utils import *

def train_val_split(part1_rootdir, dst_dir, splitAt=2421, resized=(512, 512)):
    '''resize the image to 512x512 and put it them in one folder'''
    JPGs = glob.iglob(part1_rootdir + '**/*.JPG', recursive=True)
    JPGs = [jpg for jpg in JPGs if 'Label' not in jpg]
    # assert len(JPGs) == 3021
    random.shuffle(JPGs)

    for jpg in JPGs[:splitAt]:
        img = cv2.imread(jpg)
        img = cv2.resize(img, resized)
        names = jpg.split('/')
        pref, imname = names[-2], names[-1]
        cv2.imwrite(os.path.join(dst_dir, 'train', pref + '_' + imname), img)

    for jpg in JPGs[splitAt:]:
        img = cv2.imread(jpg)
        img = cv2.resize(img, resized)
        names = jpg.split('/')
        pref, imname = names[-2], names[-1]
        cv2.imwrite(os.path.join(dst_dir, 'val', pref + '_' + imname), img)


def apply_gamma_high(x):
    result = x/255.0 ** random.uniform(0.75, 0.95) *255.0
    return np.uint8(np.clip(result, 0, 255))


def apply_gamma_low(x):
    return np.round(x/255.0 ** random.uniform(1.2, 2) *255.0).astype(np.uint8)


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
    alpha=random.uniform(1.2, 2)
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


def high_saturation(image, saturation_scale=random.uniform(1.5, 2)):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    s = cv2.multiply(s, saturation_scale)
    hsv_image = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)


def low_saturation(image, saturation_scale=random.uniform(0.5, 0.9)):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    s = cv2.multiply(s, saturation_scale)
    hsv_image = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)


class SICE(Dataset):
    def __init__(self, img_dir, transform=None):
        self.base_dir = img_dir
        self.source1_dir = img_dir +'/source1'
        print(self.source1_dir)
        self.source2_dir = img_dir +'/source2'
        print(self.source2_dir)

        self.s1 = [im_name for im_name in os.listdir(self.source1_dir)
                       if im_name.split('.')[-1].lower() in ('jpg', 'png', 'bmp')]
        self.s2 = [im_name for im_name in os.listdir(self.source2_dir)
                       if im_name.split('.')[-1].lower() in ('jpg', 'png', 'bmp')]
        self.source1 = self.s1 # +self.s2
        self.source2 = self.source1
        self.transform = transform

    def __len__(self):
        return len(self.source1)

    def __getitem__(self, idx):
        name = self.source1[idx]
        img1 = cv2.imread(os.path.join(self.source1_dir, name))
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.imread(os.path.join(self.source2_dir, name))
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        distortions = {1: gaussian_noise, 2: apply_gamma_low, 3: apply_gamma_high, 4: lower_contrast,
                       5: higher_contrast, 6: low_saturation, 7: ori}

        img1_r = np.copy(img1)
        img2_r = np.copy(img2)

        noise_func = distortions.get(1)

        H = img1.shape[0]
        W = img1.shape[1]
        x = random.randint(0, H//3-1)
        y = random.randint(0, W//3-1)
        h = random.randint(H // 3*2, H-1)
        w = random.randint(W // 3*2, W - 1)

        dist1 = random.randint(1, 7)
        dist_func1 = distortions.get(dist1)
        if random.randint(0, 5) < 3:
            img1 = noise_func(dist_func1(img1))
        else:
            img1 = dist_func1(img1)
        img1_r[x:h, y:w, :]=img1[x:h, y:w, :]

        H = img2.shape[0]
        W = img2.shape[1]
        x = random.randint(0, H//3-1)
        y = random.randint(0, W//3-1)
        h = random.randint(H // 3 * 2, H-1)
        w = random.randint(W // 3 * 2, W - 1)

        dist2 = random.randint(1, 7)
        dist_func2 = distortions.get(dist2)
        if random.randint(0, 5) < 3:
            img2 = noise_func(dist_func2(img2))
        else:
            img2 = dist_func2(img2)
        img2_r[x:h, y:w, :] = img2[x:h, y:w, :]

        if self.transform:
            seed = torch.random.seed()
            torch.random.manual_seed(seed)
            img1_r = self.transform(img1_r)
            torch.random.manual_seed(seed)
            img2_r = self.transform(img2_r)

        sample = {'name': name, 'img1': img1_r, 'img2': img2_r}
        return sample


class SICE_F_stru(Dataset):
    def __init__(self, img_dir, transform=None):
        self.base_dir = img_dir
        self.source1_dir = img_dir +'/source1'
        self.source2_dir = img_dir + '/source2'
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
        img2 = cv2.imread(os.path.join(self.source2_dir, name))
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        distortions = {1: apply_gamma_low, 2: apply_gamma_high, 3: lower_contrast, 4: low_saturation,
                       5: gaussian_noise}

        img1_r = img1
        img2_r = img2

        H = img1.shape[0]
        W = img1.shape[1]
        x = random.randint(0, H//2-1)
        y = random.randint(0, W//2-1)
        h = random.randint(H // 2, H-1)
        w = random.randint(W // 2, W - 1)

        func = distortions.get(1)

        dist1 = random.randint(1, 5)
        dist_func1 = distortions.get(dist1)
        if random.randint(0, 4)<2:
            img1 = func(dist_func1(img1))
        else:
            img1 = dist_func1(img1)
        img1_r[x:h, y:w, :]=img1[x:h, y:w, :]

        H = img2.shape[0]
        W = img2.shape[1]
        x = random.randint(0, H//2-1)
        y = random.randint(0, W//2-1)
        h = random.randint(H // 2, H-1)
        w = random.randint(W // 2, W - 1)

        dist2 = random.randint(1, 5)
        dist_func2 = distortions.get(dist2)
        if random.randint(0, 4) < 2:
            img2 = func(dist_func2(img2))
        else:
            img2 = dist_func2(img2)
        img2_r[x:h, y:w, :]=img2[x:h, y:w, :]

        if self.transform:
            seed = torch.random.seed()
            torch.random.manual_seed(seed)
            img1_r = self.transform(img1_r)
            torch.random.manual_seed(seed)
            img2_r = self.transform(img2_r)

        sample = {'name': name, 'img1': img1_r, 'img2': img2_r}
        return sample


class SICE_F_attr(Dataset):
    def __init__(self, img_dir, transform=None):
        self.s1_dir = '../dataset/train/source1/'
        self.s2_dir = '../dataset/train/source2/'

        self.s1 = [im_name for im_name in os.listdir(self.s1_dir)
                       if im_name.split('.')[-1].lower() in ('jpg', 'png', 'bmp')]
        self.s2 = [im_name for im_name in os.listdir(self.s2_dir)
                       if im_name.split('.')[-1].lower() in ('jpg', 'png', 'bmp')]
        self.transform = transform

    def __len__(self):
        return np.min([len(self.s1), len(self.s2)]) # len(self.source),

    def __getitem__(self, idx):
        img_1 = cv2.cvtColor(cv2.imread(os.path.join(self.s1_dir, self.s1[idx])), cv2.COLOR_BGR2RGB)
        img_2 = cv2.cvtColor(cv2.imread(os.path.join(self.s2_dir, self.s2[idx])), cv2.COLOR_BGR2RGB)

        height1, width1, _ = img_2.shape
        if height1 > 1000 or width1 > 800:
            img_1 = cv2.resize(img_1, (1000, 800))
            img_2 = cv2.resize(img_2, (1000, 800))

        height1, width1, _ = img_2.shape
        if height1 < 320 or width1 < 320:
            img_1 = cv2.resize(img_1, (320, 320))
            img_2 = cv2.resize(img_2, (320, 320))

        distortions = {1: apply_gamma_low, 2: low_saturation, 3: lower_contrast,
                       4: apply_gamma_high, 5: higher_contrast, 6: high_saturation}

        dist1 = random.randint(1, 6)
        dist_func1 = distortions.get(dist1)
        dist2 = random.randint(1, 6)
        dist_func2 = distortions.get(dist2)

        img_1 = dist_func1(img_1)
        img_2 = dist_func2(img_2)

        if self.transform:
            seed = torch.random.seed()
            torch.random.manual_seed(seed)
            torch.random.manual_seed(seed)
            img1 = self.transform(img_1)
            torch.random.manual_seed(seed)
            img2 = self.transform(img_2)

        sample = {'img_1': img1, 'img_2': img2}
        return sample


class SICE_TEST(Dataset):
    def __init__(self, img_dir, transform=None):
        self.base_dir = img_dir
        self.source1_dir = img_dir +'/source1'
        print(self.source1_dir)
        self.source2_dir = img_dir +'/source2'
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
        img2 = cv2.imread(os.path.join(self.source2_dir, name))
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

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