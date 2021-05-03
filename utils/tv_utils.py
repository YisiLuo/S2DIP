import torch
import scipy.io
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

dtype = torch.cuda.FloatTensor
def sam(T, H):
    assert T.ndim ==3 and T.shape == H.shape
    t1 = sum(T*H)
    t2 = sum(T*T)
    t3 = sum(H*H)
    sam_all = np.arccos(t1/np.sqrt(t2 * t3 + 1e-10))
    return sam_all.mean()

class TV_Loss(nn.Module):
    def __init__(self):
        super(TV_Loss, self).__init__()

    def forward(self,a):
        gradient_a_x = torch.abs(a[ :, :, :, :-1] - a[ :, :, :, 1:])
        gradient_a_y = torch.abs(a[ :, :, :-1, :] - a[ :, :, 1:, :])
        return torch.mean(gradient_a_x) + torch.mean(gradient_a_y)

class GTV_Loss(nn.Module):
    def __init__(self):
        super(GTV_Loss, self).__init__()

    def forward(self, a):
        gradient_a_x = torch.abs(a[:, :, :, :, :-1] - a[:, :, :, :, 1:])
        gradient_a_y = torch.abs(a[:, :, :, :-1, :] - a[:, :, :, 1:, :])
        gradient_a_z = torch.abs(a[:, :, :-1, :, :] - a[:, :, 1:, :, :])
        return 0.1*torch.mean(gradient_a_x) + 0.1*torch.mean(gradient_a_y) + 0.4*torch.mean(gradient_a_z)

class HSSTV_Loss(nn.Module):
    def __init__(self):
        super(HSSTV_Loss, self).__init__()

    def forward(self, a):
        gradient_a_x = torch.abs(a[:, :, :, :, :-1] - a[:, :, :, :, 1:])
        gradient_a_y = torch.abs(a[:, :, :, :-1, :] - a[:, :, :, 1:, :])
        gradient_a_z = torch.abs(a[:, :, :-1, :, :] - a[:, :, 1:, :, :])
        gradient_a_yz = torch.abs(gradient_a_z[:, :, :, :-1, :] - gradient_a_z[:, :, :, 1:, :])
        gradient_a_xz = torch.abs(gradient_a_z[:, :, :, :, :-1] - gradient_a_z[:, :, :, :, 1:])
        return 0.01*torch.mean(gradient_a_x) + 0.01*torch.mean(gradient_a_y) + 1*torch.mean(gradient_a_xz) + 1*torch.mean(gradient_a_yz)
        
class HSSTV_Loss_p(nn.Module):
    def __init__(self,p):
        super(HSSTV_Loss_p, self).__init__()
        self.p=p
        
    def forward(self, a):
        #print(a.type())
        gradient_a_x = torch.norm(a[:, :, :, :, :-1] - a[:, :, :, :, 1:],p=self.p)
        gradient_a_y = torch.norm(a[:, :, :, :-1, :] - a[:, :, :, 1:, :],p=self.p)
        gradient_a_z = a[:, :, :-1, :, :] - a[:, :, 1:, :, :]
        gradient_a_yz = torch.norm(gradient_a_z[:, :, :, :-1, :] - gradient_a_z[:, :, :, 1:, :],p=self.p)
        gradient_a_xz = torch.norm(gradient_a_z[:, :, :, :, :-1] - gradient_a_z[:, :, :, :, 1:],p=self.p)
        #retur = torch.norm(a[:],p=self.p)
        #print(retur)
        sol = (0.001*gradient_a_x + 0.001*gradient_a_y + 0.01*gradient_a_xz + 0.01*gradient_a_yz)
        #value = torch.norm(a,p=self.p)
        #sol=torch.atan(value)
        print(sol)
        return sol
    
class HSSTV2D_Loss(nn.Module):
    def __init__(self):
        super(HSSTV2D_Loss, self).__init__()

    def forward(self, a):
        gradient_a_x = torch.abs(a[:, :, :, :-1] - a[ :, :, :, 1:])
        gradient_a_y = torch.abs(a[:, :, :-1, :] - a[ :, :, 1:, :])
        gradient_a_z = torch.abs(a[:, :-1, :, :] - a[:, 1:, :, :])
        gradient_a_yz = torch.abs(gradient_a_z[ :, :, :-1, :] - gradient_a_z[ :, :, 1:, :])
        gradient_a_xz = torch.abs(gradient_a_z[ :, :, :, :-1] - gradient_a_z[ :, :, :, 1:])
        return 0.01*torch.mean(gradient_a_x) + 0.01*torch.mean(gradient_a_y) + torch.mean(gradient_a_xz) + torch.mean(gradient_a_yz)
        #return torch.mean(gradient_a_xz) + torch.mean(gradient_a_yz)
        
class GTV2D_Loss(nn.Module):
    def __init__(self):
        super(GTV2D_Loss, self).__init__()

    def forward(self, a):
        gradient_a_x = torch.abs(a[ :, :, :, :-1] - a[ :, :, :, 1:])
        gradient_a_y = torch.abs(a[ :, :, :-1, :] - a[ :, :, 1:, :])
        gradient_a_z = torch.abs(a[ :, :-1, :, :] - a[ :, 1:, :, :])
        return torch.mean(gradient_a_x) + torch.mean(gradient_a_y) + torch.mean(gradient_a_z)


class GaussianBlur(nn.Module):
    def __init__(self):
        super(GaussianBlur, self).__init__()
        kernel = torch.zeros(7,7)+1/49
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
 
    def forward(self, x):
        sol = torch.zeros(x.size())
        for i in range(32):
            sol[0,i,:,:] = (torch.squeeze(F.conv2d(x[:,i,:,:].unsqueeze(0), self.weight, padding=6)))[3:-3,3:-3]
        return sol
    
class Tix_Loss(nn.Module):
    def __init__(self):
        super(Tix_Loss, self).__init__()

    def forward(self, a):
        gradient_a_x = torch.pow((a[:, :, :, :, :-1] - a[:, :, :, :, 1:]),2)
        gradient_a_y = torch.pow((a[:, :, :, :-1, :] - a[:, :, :, 1:, :]),2)
        return torch.mean(gradient_a_x) + torch.mean(gradient_a_y)
    
def prepare_mask(image):
    mask_np = image
    mask_np = abs(mask_np)
    for i in range(mask_np.shape[2]):
        for j in range(mask_np.shape[0]):
            if mask_np[j,:,i].sum() != 0:# and mask_np[j,:,i].mean() < 0.6:
                mask_np[j,:,i] = 1
    mask_np[mask_np < 1] = 0
    mask_var = torch.from_numpy(mask_np).type(dtype)
    return mask_np, mask_var

def prepare_image(img,channel):
    img_np=img
    #img_np = img[:,:,:channel]
    if img_np.shape[2] < channel:
        temp = np.zeros([img_np.shape[0],img_np.shape[1],32])
        temp[:,:,0:img_np.shape[2]] = img_np
        temp[:,:,31] = img_np[:,:,30]
        img_np = temp;
    else:
        img_np = img[:,:,:channel]
    if img_np.shape[0]%32 != 0 or img_np.shape[1]%32 != 0:
        img_np = img_np[:img_np.shape[0]-img_np.shape[0]%32,:img_np.shape[1]-img_np.shape[1]%32, :]
    #img_np = np.clip(img_np,0,1)
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