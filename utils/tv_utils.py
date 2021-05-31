import torch
import scipy.io
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from skimage.measure import compare_psnr

dtype = torch.cuda.FloatTensor

def psnr3d(x,y):
    p=0
    min_n3 = min(x.shape[0],y.shape[0])
    for i in range(min_n3):
        p_he = compare_psnr(np.clip(x[i,:,:],0,1),np.clip(y[i,:,:],0,1))
        #print(p_he,i)
        p += p_he
        
    return p/min_n3

class TV_Loss(nn.Module):
    def __init__(self):
        super(TV_Loss, self).__init__()

    def forward(self,a):
        gradient_a_x = torch.abs(a[ :, :, :, :, :-1] - a[ :, :, :, :, 1:])
        gradient_a_y = torch.abs(a[ :, :, :, :-1, :] - a[ :, :, :, 1:, :])
        return gradient_a_y,gradient_a_x

class SSTV_Loss(nn.Module):
    def __init__(self):
        super(SSTV_Loss, self).__init__()

    def forward(self, a):
        gradient_a_z = torch.abs(a[:, :, :-1, :, :] - a[:, :, 1:, :, :])
        gradient_a_yz = torch.abs(gradient_a_z[:, :, :, :-1, :] - gradient_a_z[:, :, :, 1:, :])
        gradient_a_xz = torch.abs(gradient_a_z[:, :, :, :, :-1] - gradient_a_z[:, :, :, :, 1:])
        return gradient_a_yz,gradient_a_xz
    
def prepare_mask(image):
    mask_np = image
    mask_np = abs(mask_np)
    for i in range(mask_np.shape[2]):
        for j in range(mask_np.shape[0]):
            if mask_np[j,:,i].sum() != 0:
                mask_np[j,:,i] = 1
    mask_np[mask_np < 1] = 0
    mask_var = torch.from_numpy(mask_np).type(dtype)
    return mask_np, mask_var

def prepare_image(img,channel):
    img_np=img
    if img_np.shape[2] < channel:
        temp = np.zeros([img_np.shape[0],img_np.shape[1],32])
        temp[:,:,0:img_np.shape[2]] = img_np
        temp[:,:,31] = img_np[:,:,30]
        img_np = temp;
    else:
        img_np = img[:,:,:channel]
    if img_np.shape[0]%32 != 0 or img_np.shape[1]%32 != 0:
        img_np = img_np[:img_np.shape[0]-img_np.shape[0]%32,:img_np.shape[1]-img_np.shape[1]%32, :]
    img_np = img_np.transpose(2,0,1)
    img_var = Variable(torch.from_numpy(img_np).type(dtype)).cuda()
    return img_np, img_var

def prepare_noise_image(img,channel):
    img_np=img
    if img_np.shape[2] < 32:
        temp = np.zeros([img_np.shape[0],img_np.shape[1],32])
        temp[:,:,0:img_np.shape[2]] = img_np
        temp[:,:,31] = img_np[:,:,30]
        img_np = temp;
    else:
        img_np = img[:,:,:channel]
    if img_np.shape[0]%32 != 0 or img_np.shape[1]%32 != 0:
        img_np = img_np[:img_np.shape[0]-img_np.shape[0]%32,:img_np.shape[1]-img_np.shape[1]%32, :]
    img_np = img_np.transpose(2,0,1)
    img_var = Variable(torch.from_numpy(img_np).type(dtype)).cuda()
    return img_np, img_var

class permute_in(nn.Module):
    def __init__(self, dim):
        super(permute_in, self).__init__()
        self.dim = dim
    
    def forward(self, x):
        x = x.permute(2,0,1)
        x_in = x.view(1,1,self.dim,x.shape[1],x.shape[2])
        return x_in
    
class permute_out(nn.Module):
    def __init__(self, dim):
        super(permute_out, self).__init__()
        self.dim = dim
    
    def forward(self, x):
        x = x.view(self.dim,x.shape[3],x.shape[4])
        x_out = x.permute(1,2,0)
        return x_out
    
class soft(nn.Module):
    def __init__(self):
        super(soft, self).__init__()
    
    def forward(self, x, lam):
        x_abs = x.abs()-lam
        zeros = x_abs - x_abs
        n_sub = torch.max(x_abs, zeros)
        x_out = torch.mul(torch.sign(x), n_sub)
        return x_out