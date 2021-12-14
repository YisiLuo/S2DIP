import torch
import torch.nn as nn
from .common3D import *

import math
import numpy as np 
from numpy import linalg as la
import tensorflow as tf

from torch import nn
dtype = torch.cuda.FloatTensor
        
class Sep_3Dlayer_A(nn.Module):
    def __init__(self, channel_in, channel_out, stride, bias=True, pad='zero', downsample_mode='stride'):
        super(Sep_3Dlayer_A, self).__init__()
        self.downsample_mode = downsample_mode
        self.conv_1 = nn.Conv3d(channel_in, channel_out, [1,3,3], stride, bias = bias, padding = 1)
        self.conv_2 = nn.Conv3d(channel_out, channel_out, [5,1,1], stride, bias = bias, padding = (1,0,0))
        self.pool = nn.MaxPool3d(2,2)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self,x):
        d1 = self.conv_1(x)
        d1 = self.relu(d1)
        d = self.conv_2(d1)
        d = self.relu(d)
        if self.downsample_mode == 'max':
            d = self.pool(d)
        return d
    
class Sep_3Dlayer_B(nn.Module):
    def __init__(self, channel_in, channel_out, stride, bias=True, pad='zero', downsample_mode='stride'):
        super(Sep_3Dlayer_B, self).__init__()
        self.downsample_mode = downsample_mode
        self.conv_1 = nn.Conv3d(channel_in, channel_out, [1,3,3], stride, bias = bias, padding = (0,1,1))
        self.conv_2 = nn.Conv3d(channel_in, channel_out, [5,1,1], stride, bias = bias, padding = (2,0,0))
        self.pool = nn.MaxPool3d(2,2)
        self.relu = nn.ReLU()
        
    def forward(self,x):
        s = x.size()
        d1 = self.conv_1(x)#
        d1 = self.relu(d1)
        d2 = self.conv_2(x)#
        #print(d1.size(),d2.size())
        d = d1+d2
        if self.downsample_mode == 'max':
            d = self.pool(d)
        return d
        
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel,channel//reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel//reduction,channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        print(x.size())
        _, b, c, _, _ = x.size()
        y = self.avg_pool(x[0,:]).view(b, c)
        y = self.fc(y)
        y = y.view(1,b, c, 1, 1)
        return x * y.expand_as(x)

class countlayer(nn.Module):
    def __init__(self):
        super(countlayer, self).__init__()
        
    def forward(self, x):
        print('tensor size:',x.size())
        return x


#from tensorflow.python.framework import ops
        
def skip(  hs_depth,
        num_input_channels=2, num_output_channels=3, 
        num_channels_down=[16, 32, 64, 128, 128], num_channels_up=[16, 32, 64, 128, 128], num_channels_skip=[4, 4, 4, 4, 4], 
        filter_size_down=3, filter_size_up=3, filter_skip_size=1, kernel_one = [1,3,3],kernel_two = [3,1,1],
        need_sigmoid=True, need_bias=True, 
        pad='zero', upsample_mode='nearest', downsample_mode='stride', act_fun='LeakyReLU', 
        need1x1_up=False):
    """
        act_fun: Either string 'LeakyReLU|Swish|ELU|none' or module (e.g. nn.ReLU)
        pad (string): zero|reflection (default: 'zero')
        upsample_mode (string): 'nearest|bilinear' (default: 'nearest')
        downsample_mode (string): 'stride|avg|max|lanczos2' (default: 'stride')

    """
    assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)

    n_scales = len(num_channels_down) 

    if not (isinstance(upsample_mode, list) or isinstance(upsample_mode, tuple)) :
        upsample_mode   = [upsample_mode]*n_scales

    if not (isinstance(downsample_mode, list)or isinstance(downsample_mode, tuple)):
        downsample_mode   = [downsample_mode]*n_scales
    
    if not (isinstance(filter_size_down, list) or isinstance(filter_size_down, tuple)) :
        filter_size_down   = [filter_size_down]*n_scales

    if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)) :
        filter_size_up   = [filter_size_up]*n_scales

    last_scale = n_scales - 1 

    cur_depth = None

    model = nn.Sequential()
    model_tmp = model

    input_depth = num_input_channels
    for i in range(len(num_channels_down)):

        deeper = nn.Sequential()
        skip = nn.Sequential()

        if num_channels_skip[i] != 0:
            model_tmp.add(Concat(1, skip, deeper))
        else:
            model_tmp.add(deeper)
        
        model_tmp.add(bn(num_channels_skip[i] + (num_channels_up[i + 1] if i < last_scale else num_channels_down[i])))

        if num_channels_skip[i] != 0:
            skip.add(conv(input_depth, num_channels_skip[i], filter_skip_size, bias=need_bias, pad=pad))
            #skip.add(Sep_3Dlayer_B(input_depth, num_channels_skip[i], filter_skip_size, bias=need_bias, pad=pad))
            skip.add(bn(num_channels_skip[i]))
            skip.add(act(act_fun))
            

        #deeper.add(Sep_3Dlayer_B(input_depth, num_channels_down[i], stride = 1, bias=True, pad=pad))
        deeper.add(Sep_3Dlayer_A(input_depth, num_channels_down[i], stride = 1, bias=True, pad=pad))
        #deeper.add(conv(input_depth, num_channels_down[i], filter_size_down[i], 2, bias=need_bias, pad=pad, downsample_mode=downsample_mode[i]))
        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))

        #deeper.add(Sep_3Dlayer_B(num_channels_down[i], num_channels_down[i], stride = 1, bias=True, pad=pad, downsample_mode='max'))
        deeper.add(Sep_3Dlayer_A(num_channels_down[i], num_channels_down[i], stride = 1, bias=True, pad=pad, downsample_mode='max'))           
        #deeper.add(conv(num_channels_down[i], num_channels_down[i], filter_size_down[i], bias=need_bias, pad=pad))
        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))
                
        #deeper.add(countlayer())
        deeper_main = nn.Sequential()

        if i == len(num_channels_down) - 1:
            k = num_channels_down[i]
        else:
            deeper.add(deeper_main)
            k = num_channels_up[i + 1]
        
        deeper.add(nn.Upsample(scale_factor=2, mode=upsample_mode[i]))

        #model_tmp.add(Sep_3Dlayer_B(num_channels_skip[i] + k, num_channels_up[i], stride = 1, bias=True, pad=pad))
        model_tmp.add(Sep_3Dlayer_A(num_channels_skip[i] + k, num_channels_up[i], stride = 1, bias=True, pad=pad))
        #model_tmp.add(conv(num_channels_skip[i] + k, num_channels_up[i], filter_size_up[i], 1, bias=need_bias, pad=pad))
        model_tmp.add(bn(num_channels_up[i]))
        model_tmp.add(act(act_fun))


        #if need1x1_up:
         #   model_tmp.add(conv(num_channels_up[i], num_channels_up[i], 1, bias=need_bias, pad=pad))
         #   model_tmp.add(bn(num_channels_up[i]))
         #   model_tmp.add(act(act_fun))
            
        #if i == 0:
         #   deeper.add(SELayer(hs_depth))

        #if i == 1:
         #   par = hs_depth//2
         #   deeper.add(SELayer(par))
        

        input_depth = num_channels_down[i]
        model_tmp = deeper_main

    model.add(conv(num_channels_up[0], num_output_channels, 1, bias=need_bias, pad=pad))
    #model.add(Sep_3Dlayer_B(num_channels_up[0], num_output_channels,1, bias=need_bias, pad=pad))
    if need_sigmoid:
        model.add(nn.Sigmoid())

    return model
