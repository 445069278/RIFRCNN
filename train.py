import torch
from model import MyGeneratorNet
from read_data import MyDataset
from torch.autograd import Variable
from skimage import io, transform
from torchvision import transforms, utils
import torch.backends.cudnn as cudnn
import numpy as np
import cv2

model = MyGeneratorNet()
dataset = MyDataset("datasets", "train_QB") #train_QB或train_WV

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

#loss_func_1 = torch.nn.MSELoss().cuda()
loss_func_2 = torch.nn.L1Loss().cuda()

for i in range(100):
	print('第' + str(i + 1) + '轮')
	for j in range(dataset.__len__()):

		pan, ms_nir, gt_img = dataset[j]

		pan = pan.view(1, pan.size()[0], pan.size()[1], pan.size()[2])
		ms_nir = ms_nir.view(1, ms_nir.size()[0], ms_nir.size()[1], ms_nir.size()[2])
		gt_img = gt_img.view(1, gt_img.size()[0], gt_img.size()[1], gt_img.size()[2])
		pan, ms_nir, gt_img = Variable(pan).cuda(), Variable(ms_nir).cuda(), Variable(gt_img).cuda()

		#print('------------------------------------------------------------')
		#print('第' + str(i+1) + '个')
		model = model.cuda()
		net1 = model(pan, ms_nir)

		#loss_func_One = loss_func_1(net1, gt_img)
		loss_func_Two = loss_func_2(net1, gt_img)

		loss = loss_func_Two

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# print('*****************')

		if (i+1)%100 == 0:
			
			print('第' + str(i + 1) + '轮')
			print('Total_loss:' + str(loss.item()))
			net_rbgn = net1.squeeze(0)

			save_rbg = net_rbgn[[0,1,2], :, :]
			re_rbg = save_rbg.cuda().cpu()
			re_rbg = transforms.ToPILImage()(re_rbg)

			save_bgn = net_rbgn[[3,0,1], :, :]
			re_bgn = save_bgn.cuda().cpu()
			re_bgn = transforms.ToPILImage()(re_bgn)

			re_rbg.save('./datasets/results_Y/re_rbg_' + str(i) + '_' + str(j) + '.png')
			re_bgn.save('./datasets/results_Y/re_rbn_' + str(i) + '_' + str(j) + '.png')

	torch.save(model.state_dict(), './pkl_QB/parameter' + str(i+1) + '.pkl')
