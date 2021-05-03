import torch
import torch.nn as nn
import torchvision
import sys
from visdom import Visdom
import numpy as np
from PIL import Image
import PIL
import numpy as np
import xlsxwriter

import matplotlib.pyplot as plt

def crop_image(img, d=32):
    '''Make dimensions divisible by `d`'''

    new_size = (img.size[0] - img.size[0] % d, 
                img.size[1] - img.size[1] % d)

    bbox = [
            int((img.size[0] - new_size[0])/2), 
            int((img.size[1] - new_size[1])/2),
            int((img.size[0] + new_size[0])/2),
            int((img.size[1] + new_size[1])/2),
    ]

    img_cropped = img.crop(bbox)
    return img_cropped

def get_params(opt_over, net1, downsampler=None):
    '''Returns parameters that we want to optimize over.

    Args:
        opt_over: comma separated list, e.g. "net,input" or "net"
        net: network
        net_input: torch.Tensor that stores input `z`
    '''
    opt_over_list = opt_over.split(',')
    params = []
    
    for opt in opt_over_list:
    
        if opt == 'net':
            params = [x for x in net1.parameters() ]
        else:
            assert False, 'what is it?'
            
    return params

def fill_noise(x, noise_type):
    """Fills tensor `x` with noise of type `noise_type`."""
    if noise_type == 'u':
        x.uniform_()
    elif noise_type == 'n':
        x.normal_() 
    else:
        assert False

def get_noise(input_depth, method, spatial_size, noise_type='u', var=1./10):
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)
        
    if method == '2D':
        shape = [1, input_depth, spatial_size[0], spatial_size[1]] 
    elif method == 'Sep-3D' or method == '3D':
        shape = [1, 1, input_depth, spatial_size[0], spatial_size[1]]
    else:
        assert False

    net_input = torch.zeros(shape)
    
    fill_noise(net_input, noise_type)
    net_input *= var            

        
    return net_input
        
def optimize_vis(optimizer_type, parameters, closure, LR, num_iter, R = 0):
    print('Starting optimization with ADAM')
    optimizer = torch.optim.Adam(parameters, lr=LR)
    file_name = "vis.xlsx"   
    workbook = xlsxwriter.Workbook(file_name)
    worksheet = workbook.add_worksheet('sheet1')
    #viz = Visdom()
    #viz_1 = Visdom()
    #PSNR = viz.line([0.],[0.],opts = dict(title='PSNR'))
    #RE = viz_1.line([0.],[0.],opts = dict(title='Relative Error'))
    if R == 0:
        for j in range(num_iter):
            optimizer.zero_grad()
            p,re = closure(j)
            worksheet.write(j,0,p)
            worksheet.write(j,1,j)
            #viz.line([p],[j],win = PSNR,update = 'append')
            #if j >= 100:
                #viz_1.line([re],[j],win = RE,update = 'append')
            optimizer.step()
        workbook.close()
    else:
        for j in range(num_iter):
            optimizer.zero_grad()
            p,re,t = closure(j)
            if t >= 5:
                break
            #viz.line([p],[j],win = PSNR,update = 'append')
            #if j >= 100:
                #viz_1.line([re],[j],win = RE,update = 'append')
            optimizer.step()
            
            
def optimize_vis_pro(optimizer_type, parameters, closure, LR, num_iter, R = 0):
    print('Starting optimization with ADAM')
    optimizer = torch.optim.Adam(parameters, lr=LR)
    file_name = "vis_pro.xlsx"   
    workbook = xlsxwriter.Workbook(file_name)
    worksheet = workbook.add_worksheet('sheet1')
    #viz = Visdom()
    #viz_1 = Visdom()
    #PSNR = viz.line([0.],[0.],opts = dict(title='PSNR'))
    #RE = viz_1.line([0.],[0.],opts = dict(title='Relative Error'))
    if R == 0:
        for j in range(num_iter):
            optimizer.zero_grad()
            p,re = closure(j)
            worksheet.write(j,0,p)
            worksheet.write(j,1,j)
            #viz.line([p],[j],win = PSNR,update = 'append')
            #if j >= 100:
                #viz_1.line([re],[j],win = RE,update = 'append')
            optimizer.step()
        workbook.close()
    else:
        for j in range(num_iter):
            optimizer.zero_grad()
            p,re,t = closure(j)
            if t >= 5:
                break
            #viz.line([p],[j],win = PSNR,update = 'append')
            #if j >= 100:
                #viz_1.line([re],[j],win = RE,update = 'append')
            optimizer.step()
        
    
def optimize(optimizer_type, parameters, closure, LR, num_iter, R = 0):
    print('Starting optimization with ADAM')
    optimizer = torch.optim.Adam(parameters, lr=LR)
    if R == 0:
        for j in range(num_iter):
            optimizer.zero_grad()
            _,_ = closure(j)
            optimizer.step()
    else:
        for j in range(num_iter):
            optimizer.zero_grad()
            _,_,t = closure(j)
            if t >= 5:
                break
            optimizer.step()