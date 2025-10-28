import os
import argparse
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch
import torch.nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.manifold import MDS
from model import *
from utils import *
from dataset import *
import seaborn as sns
import umap
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

eps=1e-6

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0, help='device num, cuda:0')
    parser.add_argument('--testDir', type=str, default='./dataset/train/A2V_val/', help='path to test images')
    parser.add_argument('--outputDir', type=str, default='../train-jobs/', help='path to save the results')
    args = parser.parse_args()
    return args

args = parse_args()

if torch.cuda.is_available():
    device = torch.device(f'cuda:{args.device}')
    torch.cuda.manual_seed(1234)
else:
    device = torch.device('cpu')
    torch.manual_seed(1234)


model_A = A2V_Encoder()

checkpoint = torch.load('../train-jobs/ckpt/A2V_ckpt.pth', map_location=device)
epoch = checkpoint['epoch']
print("Epoch: ", epoch)
model_A.load_state_dict(checkpoint['model_A_state_dict'])
model_A.to(device)
model_A.eval()

sub_dirs=['vis']

test_loaders=[]
for d in range(len(sub_dirs)):
    test_dataset = SICE_A_eval(args.testDir+sub_dirs[d], transform=transforms.ToTensor())
    test_loaders.append(DataLoader(test_dataset, batch_size=1, shuffle=False))

os.makedirs(args.outputDir, exist_ok=True)

colors= ['r','g','b','y','c']
vector_group_num=[]
with torch.no_grad():
    for l in range(len(test_loaders)):
        vectors = []
        for i, sample in enumerate(test_loaders[l]):
            print(i)
            source = sample['img'].to(device)
            random_permutation = torch.randperm(source.size(1))
            RGB_shuffle = source[:, random_permutation, :, :]
            a1, b1, a2, b2, a3, b3, r1, r2 = model_A(RGB_shuffle)
            source_vector = torch.cat((a1, b1, a2, b2, a3, b3, r1, r2), dim=1)
            vectors.append(source_vector[0,:].detach().cpu().numpy())
        vectors = np.array(vectors)
        vector_group_num.append(vectors.shape[0])
        if l==0:
            vectors_all=vectors
        else:
            vectors_all = np.concatenate((vectors_all, vectors), axis=0)

        mat_dict = {'vector': vectors, 'centroid': np.mean(vectors, axis=0)}
        scipy.io.savemat(args.outputDir+'vis.mat', mat_dict)