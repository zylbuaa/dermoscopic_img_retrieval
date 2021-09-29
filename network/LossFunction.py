# -*- coding:utf-8 -*-
'''
@Updatingtime: 2021/9/29 11:30
@Author      : Yilan Zhang
@Filename    : LossFunction.py
@Email       : zhangyilan@buaa.edu.cn
'''

import numpy
import torch
import torch.nn as nn

class rotation_invariance_loss(nn.Module):
    def __init__(self,epsilon=0.5):
        super(rotation_invariance_loss, self).__init__()
        self.epsilon=epsilon

    def forward(self,u1,u2,u3,u4):
        u_mean = (u1 + u2 + u3 + u4) / 4.0
        loss1 = torch.mean(torch.log(1 + torch.abs(u1 - u_mean) / self.epsilon))
        loss2 = torch.mean(torch.log(1 + torch.abs(u2 - u_mean) / self.epsilon))
        loss3 = torch.mean(torch.log(1 + torch.abs(u3 - u_mean) / self.epsilon))
        loss4 = torch.mean(torch.log(1 + torch.abs(u4 - u_mean) / self.epsilon))

        loss = torch.mean(loss1 + loss2 + loss3 + loss4)
        return loss