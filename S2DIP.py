from __future__ import print_function
from models import pytorch_ssim
from utils.tv_utils import *
from torch.autograd import Variable
import matplotlib.pyplot as plt
%matplotlib inline
import os
import scipy.io
import numpy as np
from models.skipP3D import skip 
import torch
import torch.optim
import matplotlib.image as mp
from skimage.measure import compare_psnr
from utils.denoising_utils import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor
sigma = 0
sigma_ = sigma/255.

show = [25,16,28]#show band

#image load
file_name  = 'data/noise_hsi/simcase1'
mat = scipy.io.loadmat(file_name)
img_noisy = mat["Nhsi"]
img_noisy_np, img_noisy_var = prepare_noise_image(img_noisy,32)# Noisy image

file_name  = 'data/GT/simcl'
mat = scipy.io.loadmat(file_name)
img = mat["simu_indian"]
img_np, img_var = prepare_image(img,32)# Ground truth

#prepare mask for image inpainting
'''
mask_np, mask_var = prepare_mask(img_noisy_np)
mask_var = torch.from_numpy(mask_np).type(dtype)
'''

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(15,15))
ax1.imshow(torch.stack((img_var[show[0],:,:].cpu(),img_var[show[1],:,:].cpu(),img_var[show[2],:,:].cpu()),2))
ax2.imshow(torch.stack((img_noisy_var[show[0],:,:].cpu(),img_noisy_var[show[1],:,:].cpu(),img_noisy_var[show[2],:,:].cpu()),2))
plt.show()

method = 'Sep-3D'
pad = 'reflection'
OPT_OVER = 'net' # 'net,input'
reg_noise_std = 0.01 # 0 0.03 0.05 0.08
OPTIMIZER='adam' # 'LBFGS'
show_every = 50
save_every = 500
exp_weight=0.99
num_iter = 7000
input_depth = img_np.shape[0] 

#trade-off
lambda_tv = 0.3
lr = 0.0005

net = skip(input_depth, 1, 1, 
    num_channels_down = [16,32,64,128],
    num_channels_up =   [16,32,64,128],
    num_channels_skip =    [4,4,4,4],  
    filter_size_up = 3,filter_size_down = 3,  filter_skip_size=1,
    upsample_mode='trilinear', 
    need_sigmoid=False, need_bias=True, pad=pad, act_fun='LeakyReLU').type(dtype)
net_input = Variable(get_noise(input_depth, method, (img_np.shape[1], img_np.shape[2])).type(dtype).detach()).cuda()

print('Input_size: ',net_input.size())
s  = sum([np.prod(list(p.size())) for p in net.parameters()]); 
print ('Number of params: %d' % s)

#Loss
mse = torch.nn.MSELoss().type(dtype).cuda()
img_noisy_var = img_noisy_var[None, None, :].cuda()
TV = HSSTV_Loss()

net_input_saved = net_input.detach().clone()
noise = net_input.detach().clone()
last_net = None
psrn_noisy_last = 0

def closure(iter):
    global net_input, ps
    if reg_noise_std > 0:
        net_input = net_input_saved + (noise.normal_() * reg_noise_std)
    net_input = Variable(net_input).cuda()
    
    out = net(net_input)
    
    # Loss function
    total_loss = mse(out,img_noisy_var)
    total_loss += lambda_tv * TV(out)
    total_loss.backward()
    
    out_np = out.detach().cpu().squeeze().numpy()
    psnr_gt    = compare_psnr(np.clip(img_np.astype(np.float32),0,1), np.clip(out_np, 0, 1)) 
    if iter == 0:
        ps = psnr_gt
    else:
        if psnr_gt > ps:
            np.save('best_PSNR_result.npy',out_np)
            ps = psnr_gt
    print ('Iteration %05d    PSNR_gt: %f ' % (iter, psnr_gt), '\r', end='')
    if iter % show_every == 0:
        out_np = np.clip(out_np, 0, 1)
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(15,15))
        ax1.imshow(np.stack((out_np[show[0],:,:],out_np[show[1],:,:],out_np[show[2],:,:]),2))
        ax2.imshow(np.stack((img_np[show[0],:,:],img_np[show[1],:,:],img_np[show[2],:,:]),2))
        plt.show()
    return psnr_gt,0

p = get_params(OPT_OVER, net, net_input)
optimize(OPTIMIZER, p, closure, lr, num_iter, R = 0)

