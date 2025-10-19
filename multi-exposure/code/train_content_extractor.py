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

eps = 1e-6

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--device', type=int, default=0, help='device name, cuda:0')
	parser.add_argument('--experiment', default='content-extractor',
						help='prefix of outputs, e.g., experiment_best_model.pth will be saved to ckpt/')
	parser.add_argument('--baseDir', type=str, default='../dataset/', help='baseDir/train, baseDir/val will be used')
	parser.add_argument('--testDir', type=str, default='../dataset/test/', help='path to test images')
	parser.add_argument('--weights', type=float, default=25)
	parser.add_argument('--numEpoch', type=int, default=40)
	parser.add_argument('--patchsize', type=int, default=160)
	parser.add_argument('--batchsize', type=int, default=12)
	args = parser.parse_args()
	return args


def train(loaders, model_R, optimizer_R, writer, epoch, num_epochs, begin_time, begin_epoch, device, **kwargs):
	tau = 0.4
	model_R = model_R.train()

	vgg16 = models.vgg16(pretrained=True).features
	vgg16.cuda().eval()
	vgg16.to(device)
	print(f'--- Epoch {epoch + begin_epoch + 1} ---')
	total_i = len(loaders['train'])

	for i, sample in enumerate(loaders['train']):
		optimizer_R.zero_grad()
		source1_batch = sample['img1'].to(device)
		source2_batch = sample['img2'].to(device)

		s1_ex_feas = model_R(source1_batch)
		s2_ex_feas = model_R(source2_batch)

		vgg_layers = ['0']

		s1_selected_feas_ori = []
		x = source1_batch
		s1_gaussian = x
		for name, layer in vgg16._modules.items():
			x = layer(x)
			if name in vgg_layers:
				s1_selected_feas_ori.append(x)

		s2_selected_feas_ori = []
		x = source2_batch

		s2_gaussian = x
		for name, layer in vgg16._modules.items():
			x = layer(x)
			if name in vgg_layers:
				s2_selected_feas_ori.append(x)
		s1_selected_feas = []
		s2_selected_feas = []
		for fea_num in range(len(vgg_layers)):
			s1_selected_fea = nn.functional.leaky_relu(s1_selected_feas_ori[fea_num])
			s2_selected_fea = nn.functional.leaky_relu(s2_selected_feas_ori[fea_num])
			s1_selected_feas.append(s1_selected_fea)
			s2_selected_feas.append(s2_selected_fea)

		'''contrastive learning'''
		all_f1_samples = torch.cat((s1_ex_feas, s2_ex_feas), 0)
		N = source1_batch.shape[0]

		crit = nn.MSELoss()
		crit_l1 = nn.L1Loss()
		loss_contrastive = 0.5 * contrastive_loss(all_f1_samples, N, tau)
		loss_sim = 0

		s1_selected_fea = s1_selected_feas[0]
		s2_selected_fea = s2_selected_feas[0]

		s1_selected_fea_GF = guided_filter(s1_selected_fea, s2_selected_fea)
		s2_selected_fea_GF = guided_filter(s2_selected_fea, s1_selected_fea)

		pooling1 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
		s1_ex_feas_d1 = pooling1(s1_ex_feas)
		s2_ex_feas_d1 = pooling1(s2_ex_feas)

		pooling2 = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
		s1_ex_feas_d2 = pooling2(s1_ex_feas)
		s2_ex_feas_d2 = pooling2(s2_ex_feas)
		s1_selected_fea_d2 = pooling2(s1_selected_fea)
		s2_selected_fea_d2 = pooling2(s2_selected_fea)

		pooling3 = nn.AvgPool2d(kernel_size=7, stride=1, padding=3)
		s1_ex_feas_d3 = pooling3(s1_ex_feas)
		s2_ex_feas_d3 = pooling3(s2_ex_feas)

		loss_feature = crit(s1_selected_fea_GF, s1_ex_feas) + crit(s2_selected_fea_GF, s2_ex_feas)

		loss_sim = crit(s1_ex_feas, s2_ex_feas) * 0.2 + crit(s1_ex_feas_d1, s2_ex_feas_d1) * 0.8

		loss = loss_feature + 0.00001 * loss_contrastive + 0.5 * loss_sim

		loss.backward()
		optimizer_R.step()

		timeElapsed = datetime.now() - begin_time

		if (i +1) % 5 == 0:
			print('Content Extractor: Epoch: [%d/%d], Iter: [%d/%d], Loss: %.5f, Time: ' % (
			begin_epoch + epoch + 1, num_epochs, i + 1, total_i, loss.item()), timeElapsed)
			step = i + 1 + total_i * (begin_epoch + epoch)

			writer.add_scalar(tag="loss", scalar_value=loss, global_step=step)
			writer.add_scalar(tag="loss_contrastive", scalar_value=loss_contrastive, global_step=step)
			writer.add_scalar(tag="loss_feature", scalar_value=loss_feature, global_step=step)
			writer.add_scalar(tag="loss_sim", scalar_value=loss_sim, global_step=step)

			writer.add_image("source1", torchvision.utils.make_grid(source1_batch[0:3, :, :, :]), global_step=step)
			writer.add_image("source2", torchvision.utils.make_grid(source2_batch[0:3, :, :, :]), global_step=step)
			writer.add_image("source1_gaussian", torchvision.utils.make_grid(s1_gaussian[0:3, :, :, :]), global_step=step)
			writer.add_image("source2_gaussian", torchvision.utils.make_grid(s2_gaussian[0:3, :, :, :]), global_step=step)
			p=np.random.randint(0, 21)
			writer.add_image("s1_sel_feas", torchvision.utils.make_grid(s1_selected_fea[0:3, p:p+3, :, :]),
							 global_step=step)
			writer.add_image("s2_sel_feas", torchvision.utils.make_grid(s2_selected_fea[0:3, p:p+3, :, :]),
							 global_step=step)
			writer.add_image("s1_sel_feas_GF", torchvision.utils.make_grid(s1_selected_fea_GF[0:3, p:p+3, :, :]),
							 global_step=step)
			writer.add_image("s2_sel_feas_GF", torchvision.utils.make_grid(s2_selected_fea_GF[0:3, p:p+3, :, :]),
							 global_step=step)
			writer.add_image("s1_ex_feas", torchvision.utils.make_grid(s1_ex_feas[0:3, p:p+3, :, :]), global_step=step)
			writer.add_image("s2_ex_feas", torchvision.utils.make_grid(s2_ex_feas[0:3, p:p+3, :, :]), global_step=step)
			writer.add_image("a_normal", torchvision.utils.make_grid(source1_batch[0:3, :, :, :]), global_step=step)

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
trans_crop = transforms.RandomCrop(args.patchsize, padding=None, pad_if_needed=False, fill=0, padding_mode='constant')
trans_compose = transforms.Compose([trans_to_tensor, trans_crop])
train_dataset = SICE(train_dir, transform=trans_compose)

trainloader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, pin_memory=True)

loaders = {'train': trainloader}

hp = dict(lr=1e-4, wd=0, lr_decay_factor=0.998)

model_R = Structure_Encoder()
model_R.to(device)

optimizer_R = optim.Adam(model_R.parameters(), lr=hp['lr'], weight_decay=hp['wd'])

experiment = args.experiment

writer = SummaryWriter(log_dir='../train-jobs/log/'+args.experiment)
num_epochs = args.numEpoch
checkpoint_path = '../train-jobs/ckpt/' + args.experiment + '_ckpt.pth'

if os.path.exists(checkpoint_path):
	print(f'---Continue Training---')
	checkpoint = torch.load(checkpoint_path, map_location=device)
	model_R.load_state_dict(checkpoint['model_R_state_dict'])
	begin_epoch = checkpoint['epoch']
	print("begin epoch: ", begin_epoch + 1)
else:
	begin_epoch = 0

print(f'[START TRAINING JOB] -{experiment} on {datetime.now().strftime("%b %d %Y %H:%M:%S")}')
begin_time = datetime.now()

for epoch in range(num_epochs - begin_epoch):
	train(loaders, model_R, optimizer_R, writer, epoch, num_epochs, begin_time, begin_epoch, device)
	state_dict = {'model_R_state_dict': model_R.state_dict(), 'epoch': epoch + 1 + begin_epoch}
	torch.save(state_dict, checkpoint_path)
