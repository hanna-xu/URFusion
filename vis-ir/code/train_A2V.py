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

eps=1e-6

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--device', type=int, default=3, help='device name, cuda:0')
	parser.add_argument('--experiment', default='A2V',
						help='prefix of outputs, e.g., experiment_best_model.pth will be saved to ckpt/')
	parser.add_argument('--fusion_ckpt', default='content-fusion', help='prefix of fusion_ckpt, e.g., fusion_model.pth will be saved to ckpt/')
	parser.add_argument('--A2V_ckpt', default='A2V', help='prefix of A2V_ckpt, e.g., A2V_model.pth will be saved to ckpt/')
	parser.add_argument('--baseDir', type=str, default='../dataset/', help='baseDir/train, baseDir/val will be used')
	parser.add_argument('--testDir', type=str, default='../dataset/test/', help='path to test images')
	parser.add_argument('--numEpoch', type=int, default=30)
	parser.add_argument('--patchsize', type=int, default=280)
	parser.add_argument('--batchsize', type=int, default=8)
	args = parser.parse_args()
	return args

def train(loaders, model_A, model_F, optimizer_A, writer, epoch, num_epochs, begin_time, begin_epoch, device):
	model_F = model_F.eval()
	model_A = model_A.train()

	print(f'--- Epoch {epoch + begin_epoch + 1} ---')
	total_i = len(loaders['train'])

	for i, sample in enumerate(loaders['train']):
		optimizer_A.zero_grad()
		source1_batch = sample['img'].to(device)

		source2_value = torch.from_numpy(np.random.uniform(0, 1, size=(source1_batch.size(0), 1, 1, 1))).to(device)
		source2_value = source2_value.repeat(1, 3, source1_batch.size(2), source1_batch.size(3))
		source2_batch = torch.ones_like(source1_batch, dtype=torch.float) * source2_value
		source2_batch = source2_batch.to(torch.float32)

		for ba in range(source1_batch.size(0)):
			random_permutation = torch.randperm(source1_batch.size(1))
			RGB_shuffle = source1_batch[ba:ba+1, random_permutation, :, :]
			if ba==0:
				source_RGB_shuffle = RGB_shuffle
			else:
				source_RGB_shuffle = torch.cat((source_RGB_shuffle, RGB_shuffle), dim=0)

		s_a1, s_b1, s_a2, s_b2, s_a3, s_b3, s_r1, s_r2 = model_A(source_RGB_shuffle)

		fused_batch, middle_fused = model_F(torch.cat((source1_batch, source2_batch), 1), s_a1, s_b1, s_a2, s_b2, s_a3, s_b3, s_r1, s_r2, modulation = True)
		fused_batch_wo_modulation, _ = model_F(torch.cat((source1_batch, source2_batch), 1), s_a1, s_b1, s_a2, s_b2, s_a3, s_b3, s_r1, s_r2, modulation=False)

		crit_l2 = nn.MSELoss()
		crit_l1 = nn.L1Loss()

		loss_middle = crit_l2(source1_batch, middle_fused)
		loss_A2V = crit_l2(source1_batch, fused_batch)
		loss = loss_middle + loss_A2V * 2

		loss.backward()
		optimizer_A.step()

		timeElapsed = datetime.now() - begin_time

		if (i + 1) % 5 == 0:
			print(
				'Epoch: [%d/%d], Iter: [%d/%d], Loss: %.5f, Time: ' % (epoch + 1, num_epochs, i + 1, total_i, loss.item()),
				timeElapsed)
			step = i + 1 + total_i * (begin_epoch + epoch)
			writer.add_scalar(tag="loss", scalar_value=loss, global_step=step)
			writer.add_scalar(tag="loss_A2V", scalar_value=loss_A2V, global_step=step)
			writer.add_scalar(tag="loss_middle", scalar_value=loss_middle, global_step=step)
			writer.add_image("source1", torchvision.utils.make_grid(source1_batch[0:3, :, :, :]), global_step=step)
			writer.add_image("source2", torchvision.utils.make_grid(source2_batch[0:3, :, :, :]), global_step=step)
			writer.add_image("fused_img", torchvision.utils.make_grid(fused_batch[0:3, :, :, :]), global_step=step)
			writer.add_image("fused_img_wo_modulation", torchvision.utils.make_grid(fused_batch_wo_modulation[0:3, :, :, :]), global_step=step)
			writer.add_image("a", torchvision.utils.make_grid(source1_batch[0:3, :, :, :]), global_step=step)
		prev = datetime.now()

args = parse_args()

if torch.cuda.is_available():
	device = torch.device(f'cuda:{args.device}')
	torch.cuda.manual_seed(1234)

else:
	device_stru = torch.device('cpu')
	torch.manual_seed(1234)

basedir = args.baseDir
train_dir = os.path.join(basedir, 'train')
val_dir = os.path.join(basedir, 'val')

trans_to_tensor = transforms.ToTensor()
trans_crop = transforms.RandomCrop(args.patchsize, padding = None, pad_if_needed = False, fill = 0, padding_mode ='constant')
trans_compose = transforms.Compose([trans_to_tensor, trans_crop])
train_dataset = SICE_F_attr(train_dir, transform=trans_compose)
trainloader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, pin_memory=True)

loaders = {'train': trainloader}
hp = dict(lr=1e-4, wd=0, lr_decay_factor=0.998)

model_A = A2V_Encoder()
model_A.to(device)
model_F = FusionNet()
model_F.to(device)

all_parameters = itertools.chain(model_A.parameters(), model_F.parameters())
optimizer_A = optim.Adam(model_A.parameters(), lr=hp['lr'], weight_decay=hp['wd'])

experiment = args.experiment

writer = SummaryWriter(log_dir='../train-jobs/log/' + args.experiment)
loss_history, best_loss = [], float('inf')
num_epochs = args.numEpoch
checkpoint_path = '../train-jobs/ckpt/'+ args.experiment + '_ckpt.pth'

checkpoint_fusion = torch.load('../train-jobs/ckpt/'+args.fusion_ckpt+'_ckpt.pth', map_location = device)
model_F.load_state_dict(checkpoint_fusion['model_F_state_dict'])

print("model F: ", checkpoint_fusion['epoch'])

if os.path.exists(checkpoint_path):
	print(f'---Continue Training---')
	checkpoint = torch.load('../train-jobs/ckpt/' + args.experiment + '_ckpt.pth', map_location=device)
	model_A.load_state_dict(checkpoint['model_A_state_dict'])
	begin_epoch = checkpoint['epoch']
	print("begin epoch: ", begin_epoch + 1)
else:
	begin_epoch = 0

print(f'[START TRAINING JOB] -{experiment} on {datetime.now().strftime("%b %d %Y %H:%M:%S")}')
begin_time = datetime.now()

for epoch in range(num_epochs - begin_epoch):
	train(loaders, model_A, model_F, optimizer_A, writer, epoch, num_epochs, begin_time, begin_epoch, device)
	state_dict = {'model_F_state_dict': model_F.state_dict(), 'model_A_state_dict': model_A.state_dict(), 'epoch': epoch + 1 + begin_epoch}
	torch.save(state_dict, checkpoint_path)
