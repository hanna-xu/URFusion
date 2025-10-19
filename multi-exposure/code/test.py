import os
import argparse

# import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import torch
import torch.nn
# import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
import time
from model import *
from utils import *
from dataset import *

eps=1e-6

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0, help='device num, cuda:0')
    parser.add_argument('--testDir', type=str, default='../dataset/test/', help='path to test images')
    parser.add_argument('--task', type=str, default='ue+oe', help='path to test images')
    parser.add_argument('--content_fusion_ckpt', type=str, default='content-fusion', help='path to *_best_model.pth')
    parser.add_argument('--A2V_ckpt', type=str, default='A2V', help='path to *_best_model.pth')
    parser.add_argument('--outputDir', type=str, default='../results/', help='path to save the results')
    args = parser.parse_args()
    return args

args = parse_args()

if torch.cuda.is_available():
    device = torch.device(f'cuda:{args.device}')
    torch.cuda.manual_seed(1234)
else:
    device = torch.device('cpu')
    torch.manual_seed(1234)

model_F = FusionNet()
model_A = A2V_Encoder()
model_F.to(device)
model_F.eval()
model_A.to(device)
model_A.eval()

fusion_checkpoint = torch.load('../train-jobs/ckpt/' + args.content_fusion_ckpt + '_ckpt.pth', map_location=device)
A2V_checkpoint = torch.load('../train-jobs/ckpt/' + args.A2V_ckpt + '_ckpt.pth', map_location=device)
fusion_epoch = fusion_checkpoint['epoch']
A2V_epoch = A2V_checkpoint['epoch']
print("fusion epoch: ", fusion_epoch)
print("A2V epoch: ", A2V_epoch)
model_F.load_state_dict(fusion_checkpoint['model_F_state_dict'])
model_A.load_state_dict(A2V_checkpoint['model_A_state_dict'])

test_dataset = SICE_TEST(args.testDir + args.task, transform=transforms.ToTensor())
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

os.makedirs(args.outputDir + args.task, exist_ok=True)

with torch.no_grad():
    vector_data = scipy.io.loadmat('../train-jobs/normal_img.mat')
    centroid = vector_data['centroid']

    '''guide with statistical values'''
    centroid = torch.tensor(centroid).to(device)
    a1 = centroid[:, 0:1]
    b1 = centroid[:, 1:2]
    a2 = centroid[:, 2:3]
    b2 = centroid[:, 3:4]
    a3 = centroid[:, 4:5]
    b3 = centroid[:, 5:6]
    r1 = centroid[:, 6:7]
    r2 = centroid[:, 7:8]

    '''guide with an image'''
    # img_normal = cv2.imread('../dataset/test/ue+oe/source2/10.png')
    # img_normal = cv2.cvtColor(img_normal, cv2.COLOR_BGR2RGB)
    # transform = transforms.ToTensor()
    # img_normal = transform(img_normal).to(device).unsqueeze(0)
    # a1, b1, a2, b2, a3, b3, r1, r2 = model_A(img_normal)

    Time=[]

    for i, sample in enumerate(test_loader):
        begin=time.time()
        names = sample['name']
        source1 = sample['img1'].to(device)
        source2 = sample['img2'].to(device)
        fused_img, _ = model_F(torch.cat((source1, source2), 1), a1, b1, a2, b2,a3, b3, r1, r2, modulation=True)
        end=time.time()
        Time.append(end-begin)
        params = sum(p.numel() for p in model_F.parameters() if p.requires_grad)
        print(f"参数量: {params}")
        for name, enhanced in zip(names, fused_img):
            print(name)
            torchvision.utils.save_image(fused_img, args.outputDir + args.task + '/' + name)

    print("average time: ", np.mean(Time))
