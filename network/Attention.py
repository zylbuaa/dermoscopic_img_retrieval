# -*- coding:utf-8 -*-
'''
@Updatingtime: 2021/9/29 10:58
@Author      : Yilan Zhang
@Filename    : Attention.py
@Email       : zhangyilan@buaa.edu.cn
'''

import torch
import torch.nn as nn

class HDCSA(nn.Module):
    # reduction ratio, set r=16
    def __init__(self, gate_channel, reduction_ratio=16,dilation_conv_num=3,hybrid_dilation_val=(1,2,5),kernel_size=7):
        super(HDCSA,self).__init__()
        self.gate_s=nn.Sequential()
        # 1x1 + (3x3)*2 + 1x1
        self.gate_s.add_module('gate_s_conv_reduce0',
                               nn.Conv2d(gate_channel, gate_channel // reduction_ratio, kernel_size=1))
        self.gate_s.add_module('gate_s_bn_reduce0', nn.BatchNorm2d(gate_channel // reduction_ratio))
        self.gate_s.add_module('gate_s_relu_reduce0', nn.ReLU())
        # avoid gridding effect, set dilation rate [1,2,5]
        for i in range(dilation_conv_num):
            self.gate_s.add_module('gate_s_conv_di_%d' % i,
                                   nn.Conv2d(gate_channel // reduction_ratio, gate_channel // reduction_ratio,
                                             kernel_size=3, padding=hybrid_dilation_val[i], dilation=hybrid_dilation_val[i]))
            self.gate_s.add_module('gate_s_bn_di_%d' % i, nn.BatchNorm2d(gate_channel // reduction_ratio))
            self.gate_s.add_module('gate_s_relu_di_%d' % i, nn.ReLU())
        self.gate_s.add_module('gate_s_conv_final',
                               nn.Conv2d(gate_channel // reduction_ratio, 1, kernel_size=1))  # 1×H×W

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(3, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, in_tensor):
        MS=self.gate_s(in_tensor)

        avg_out=torch.mean(in_tensor,dim=1,keepdim=True) # mean pooling

        max_out, _ =torch.max(in_tensor,dim=1,keepdim=True) # max pooling

        x=torch.cat([avg_out,max_out,MS],dim=1)
        x = self.conv1(x)

        return self.sigmoid(x)