#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function
from models import pytorch_ssim
from utils.tv_utils import *
from torch.autograd import Variable
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import os
import scipy.io
import numpy as np
#from models.skipP3D import skip 
from models.skip import skip
import torch
import torch.optim
import matplotlib.image as mp
from utils.denoising_utils import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor
sigma = 0
sigma_ = sigma/255.

soft_thres = soft()
show = [5,15,30]#show band
data = 'wdc'
c = '5'
file_name  = 'data_HSI/'+data+'case'+c

mat = scipy.io.loadmat(file_name)
img_noisy = mat["Nhsi"]
img_noisy_np, img_noisy_var = prepare_noise_image(img_noisy,32)# Noisy image

img_var = img_noisy_var
mask = torch.ones([img_noisy_var.shape[0],img_noisy_var.shape[1],img_noisy_var.shape[2]]).type(dtype)
mask[img_noisy_var == 0] = 0
mask_np, mask_var = prepare_mask(img_noisy_np)
mask_var = torch.from_numpy(mask_np).type(dtype)

file_name  = 'data_HSI/'+data+'gt' 

mat = scipy.io.loadmat(file_name)
img = mat["Ohsi"]
img_np, img_var = prepare_image(img,32)# Ground truth

print('noisy_PSNR:',psnr3d(img_np,img_noisy_np))

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(15,15))
ax1.imshow(torch.stack((img_var[show[0],:,:].cpu(),img_var[show[1],:,:].cpu(),img_var[show[2],:,:].cpu()),2))
ax2.imshow(torch.stack((img_noisy_var[show[0],:,:].cpu(),img_noisy_var[show[1],:,:].cpu(),img_noisy_var[show[2],:,:].cpu()),2))
plt.show()

method = '2D'
pad = 'reflection'
OPT_OVER = 'net' 
reg_noise_std = 0.01 
OPTIMIZER='adam' 
show_every = 50
save_every = 500
exp_weight=0.99
num_iter = 7000
input_depth = img_noisy_np.shape[0] 
lr = 0.001

net = skip(input_depth, input_depth, img_noisy_np.shape[0],  
       num_channels_down = [128,128,128,128, 128],
       num_channels_up =   [128,128,128,128, 128],
       num_channels_skip =    [4]*5,  
       filter_size_up = 3,filter_size_down = 3,  filter_skip_size=1,
       upsample_mode='bilinear', 
       need_sigmoid=False, need_bias=True, pad=pad, act_fun='LeakyReLU').type(dtype)

net_input = Variable(get_noise(input_depth, method, (img_noisy_np.shape[1], img_noisy_np.shape[2])).type(dtype).detach()).cuda()

print('Input_size: ',net_input.size())
s  = sum([np.prod(list(p.size())) for p in net.parameters()]); 
print ('Number of params: %d' % s)

img_noisy_var = img_noisy_var[None, None, :].cuda()

TV = TV_Loss()
SSTV = SSTV_Loss()

net_input_saved = net_input.detach().clone()
noise = net_input.detach().clone()
last_net = None
psrn_noisy_last = 0

mu = 0.13
alpha_3 = 0.01
thres = 2 * alpha_3

thres_tv = 0.1
thres_sstv = 0.1

def closure(iter):
    global net_input, ps
    if reg_noise_std > 0:
        net_input = net_input_saved + (noise.normal_() * reg_noise_std)
    net_input = Variable(net_input).cuda()

    out_ = net(net_input)
    out_ = out_[None, :]

    D_x_,D_y_ = TV(out_)
    D_xz_, D_yz_ = SSTV(out_)
    D_x = D_x_.clone().detach()
    D_y = D_y_.clone().detach()
    D_xz = D_xz_.clone().detach()
    D_yz = D_yz_.clone().detach()
    out = out_.clone().detach()

    if iter == 0:
        global D_1,D_2,D_3,D_4,D_5,V_1,V_2,V_3,V_4,V_5,S,mu,thres,thres_tv,thres_sstv
        D_2 = torch.zeros([img_noisy_var.shape[0],img_noisy_var.shape[1],img_noisy_var.shape[2],
                           img_noisy_var.shape[3]-1,img_noisy_var.shape[4]]).type(dtype)
        D_3 = torch.zeros([img_noisy_var.shape[0],img_noisy_var.shape[1],img_noisy_var.shape[2],
                           img_noisy_var.shape[3],img_noisy_var.shape[4]-1]).type(dtype)
        D_4 = torch.zeros([img_noisy_var.shape[0],img_noisy_var.shape[1],img_noisy_var.shape[2]-1,
                           img_noisy_var.shape[3]-1,img_noisy_var.shape[4]]).type(dtype)
        D_5 = torch.zeros([img_noisy_var.shape[0],img_noisy_var.shape[1],img_noisy_var.shape[2]-1,
                           img_noisy_var.shape[3],img_noisy_var.shape[4]-1]).type(dtype)

        V_2 = D_x.type(dtype)
        V_3 = D_y.type(dtype)
        V_4 = D_xz.type(dtype)
        V_5 = D_yz.type(dtype)

        S = (img_noisy_var-out).type(dtype)

    S = soft_thres(img_noisy_var-out, thres)

    V_2 = soft_thres(D_x + D_2 / mu, thres_tv)
    V_3 = soft_thres(D_y + D_3 / mu, thres_tv)

    V_4 = soft_thres(D_xz + D_4 / mu,thres_sstv)
    V_5 = soft_thres(D_yz + D_5 / mu,thres_sstv)

    total_loss = mu/2 * torch.norm(D_x_-(V_2-D_2/mu),2)
    total_loss += mu/2 * torch.norm(D_y_-(V_3-D_3/mu),2)
    total_loss += 10*mu/2 * torch.norm(D_xz_-(V_4-D_4/mu),2)
    total_loss += 10*mu/2 * torch.norm(D_yz_-(V_5-D_5/mu),2)
    total_loss += torch.norm(img_noisy_var*mask-out_*mask-S,2)
    
    total_loss.backward()

    D_2 = (D_2 + mu * (D_x  - V_2)).clone().detach()
    D_3 = (D_3 + mu * (D_y  - V_3)).clone().detach()
    D_4 = (D_4 + mu * (D_xz  - V_4)).clone().detach()
    D_5 = (D_5 + mu * (D_yz  - V_5)).clone().detach()

    out_np = out.detach().cpu().squeeze().numpy()
    psnr_gt    = psnr3d(np.clip(img_np.astype(np.float32),0,1), np.clip(out_np, 0, 1)) 
    print ('Iteration %05d    PSNR_gt: %f ' % (iter, psnr_gt), '\r', end='')
    if iter % show_every == 0:
        out_np = np.clip(out_np, 0, 1)
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(15,15))
        ax1.imshow(np.stack((out_np[show[0],:,:],out_np[show[1],:,:],out_np[show[2],:,:]),2))
        ax2.imshow(np.stack((img_np[show[0],:,:],img_np[show[1],:,:],img_np[show[2],:,:]),2))
        plt.show()
    return psnr_gt,0

p = get_params(OPT_OVER, net, net_input)
net_input.requires_grad = True
p += [net_input]
optimize(OPTIMIZER, p, closure, lr, num_iter, R = 0)


# In[ ]:




