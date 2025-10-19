import os
import sys
import time
import shutil
import argparse
from datetime import datetime
from subprocess import call
import itertools
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
import torchvision
from dataset import *
from utils import *
from model import *
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import random
import torch.nn.functional as F
# from sklearn.cluster import KMeans

eps=1e-6

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--device', type=int, default=0)
	parser.add_argument('--experiment', default='content-fusion',
						help='prefix of outputs, e.g., experiment_best_model.pth will be saved to ckpt/')
	parser.add_argument('--content_ckpt', default='content-extractor', help='prefix of content-extractor_ckpt, e.g., content-extractor_ckpt.pth will be saved to ckpt/')
	parser.add_argument('--baseDir', type=str, default='../dataset/', help='baseDir/train, baseDir/val will be used')
	parser.add_argument('--testDir', type=str, default='../dataset/test/', help='path to test images')
	parser.add_argument('--numEpoch', type=int, default=20)
	parser.add_argument('--patchsize', type=int, default=160)
	parser.add_argument('--batchsize', type=int, default=12)
	args = parser.parse_args()
	return args

def train(loaders, model_R, model_A, model_F, optimizer, writer, epoch, num_epochs, begin_time, begin_epoch, device):
	tau = 0.25
	model_R = model_R.eval()
	model_A = model_A.eval()

	vgg16 = models.vgg16(pretrained=True).features
	vgg16.cuda().eval()
	vgg16.to(device)

	print(f'--- Epoch {epoch + begin_epoch + 1} ---')
	total_i = len(loaders['train'])

	for i, sample in enumerate(loaders['train']):
		source1_batch = sample['img1'].to(device)
		source2_batch = sample['img2'].to(device)

		batchsize = source1_batch.size(0)
		patchsize = source1_batch.size(2)

		optimizer.zero_grad()
		a1, b1, a2, b2, a3, b3, r1, r2 = model_A(source1_batch)
		fused_batch,_ = model_F(torch.cat((source1_batch, source2_batch), 1), a1, b1,a2, b2, a3, b3, r1, r2, modulation=False)

		s1_stru_feas = model_R(source1_batch)
		s2_stru_feas = model_R(source2_batch)

		vgg_layers = ['0']
		x = fused_batch

		for name, layer in vgg16._modules.items():
			x = layer(x)
			if name in vgg_layers:
				f_stru_feas = x

		f_stru_feas = nn.functional.leaky_relu(f_stru_feas)

		crit_l2 = nn.MSELoss()
		crit_l1 = nn.L1Loss()

		s1_fea_grad = gradient_operator(s1_stru_feas)
		s2_fea_grad = gradient_operator(s2_stru_feas)
		fea_grad_max_mask = (torch.abs(s1_fea_grad)>torch.abs(s2_fea_grad)).float()
		ex_fuse_fea = s1_stru_feas * fea_grad_max_mask + s2_stru_feas * (1-fea_grad_max_mask)

		loss_structure = crit_l1(f_stru_feas, ex_fuse_fea)
		loss = loss_structure

		loss.backward(retain_graph=True)
		optimizer.step()

		timeElapsed = datetime.now() - begin_time

		if (i + 1) % 5 == 0:
			print(
				'Content Fusion: Epoch: [%d/%d], Iter: [%d/%d], Loss: %.5f, Time: ' % (begin_epoch + epoch + 1, num_epochs, i + 1, total_i, loss.item()),
				timeElapsed)
			step = i + 1 + total_i * (begin_epoch + epoch)
			writer.add_scalar(tag="loss", scalar_value=loss, global_step=step)
			writer.add_scalar(tag="loss_structure", scalar_value=loss_structure, global_step=step)
			writer.add_image("source1", torchvision.utils.make_grid(source1_batch[0:3, :, :, :]), global_step=step)
			writer.add_image("source2", torchvision.utils.make_grid(source2_batch[0:3, :, :, :]), global_step=step)
			p = np.random.randint(0, 61)
			writer.add_image("fea_ex_fuse", torchvision.utils.make_grid(ex_fuse_fea[0:3, p:p+3, :, :]), global_step=step)
			writer.add_image("fea_vgg", torchvision.utils.make_grid(f_stru_feas[0:3, p:p+3, :, :]), global_step=step)
			writer.add_image("fused_img", torchvision.utils.make_grid(fused_batch[0:3, :, :, :] ** 0.8), global_step=step)
			writer.add_image("a", torchvision.utils.make_grid(source1_batch[0:3, :, :, :]), global_step=step)
		prev = datetime.now()

args = parse_args()

if torch.cuda.is_available():
	device = torch.device(f'cuda:{args.device}')
	torch.cuda.manual_seed(1234)
else:
	device = torch.device('cpu')
	torch.manual_seed(1234)

basedir = args.baseDir
train_dir = os.path.join(basedir, 'train')
val_dir = os.path.join(basedir, 'val')

trans_to_tensor = transforms.ToTensor()
trans_crop = transforms.RandomCrop(args.patchsize, padding = None, pad_if_needed = False, fill = 0, padding_mode ='constant')
trans_compose = transforms.Compose([trans_to_tensor, trans_crop])
train_dataset = SICE_F_stru(train_dir, transform=trans_compose)
trainloader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, pin_memory=True)


loaders = {'train': trainloader}
hp = dict(lr=1e-4, wd=0, lr_decay_factor=0.999)

model_R = Structure_Encoder()
model_R.to(device)
model_A = A2V_Encoder()
model_A.to(device)
model_F = FusionNet()
model_F.to(device)

all_parameters = itertools.chain(model_R.parameters(), model_A.parameters(), model_F.parameters())
optimizer = optim.Adam(model_F.parameters(), lr=hp['lr'], weight_decay=hp['wd'])

experiment = args.experiment

ckpt_dir = '../train-jobs/ckpt'
writer = SummaryWriter(log_dir='../train-jobs/log/' + args.experiment)
loss_history, best_loss = [], float('inf')
num_epochs = args.numEpoch
checkpoint_path = '../train-jobs/ckpt/'+ args.experiment

checkpoint_stru = torch.load('../train-jobs/ckpt/' + args.content_ckpt + '_ckpt.pth', map_location = device)
model_R.load_state_dict(checkpoint_stru['model_R_state_dict'])
print("model R: ", checkpoint_stru['epoch'])

print(os.path.exists(checkpoint_path + '_ckpt.pth'))

if os.path.exists(checkpoint_path + '_ckpt.pth'):
	print(f'---Continue Training---')
	checkpoint = torch.load('../train-jobs/ckpt/' + args.experiment + '_ckpt.pth', map_location=device)
	model_F.load_state_dict(checkpoint['model_F_state_dict'])
	begin_epoch = checkpoint['epoch']
	print("begin epoch: ", begin_epoch + 1)
else:
	begin_epoch = 0

print(f'[START TRAINING JOB] -{experiment} on {datetime.now().strftime("%b %d %Y %H:%M:%S")}')
begin_time = datetime.now()

for epoch in range(num_epochs - begin_epoch):
	train(loaders, model_R, model_A, model_F, optimizer, writer, epoch, num_epochs, begin_time, begin_epoch, device)
	state_dict = {'model_F_state_dict': model_F.state_dict(), 'epoch': epoch + 1 + begin_epoch}
	torch.save(state_dict, checkpoint_path + '_ckpt.pth')
