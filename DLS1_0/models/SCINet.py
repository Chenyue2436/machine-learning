import math
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
import torch
import argparse
import numpy as np

class Splitting(nn.Module):
    def __init__(self):
        super(Splitting, self).__init__()

    def even(self, x):
        return x[:, ::2, :]

    def odd(self, x):
        return x[:, 1::2, :]

    def forward(self, x):
        return (self.even(x), self.odd(x))

class Interactor(nn.Module):
    def __init__(self, in_planes, kernel = 5, dropout = 0.5, groups = 1, hidden_size = 1):
        super(Interactor, self).__init__()
        self.kernel_size = kernel
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.groups = groups
        if self.kernel_size == 1:
            pad_l, pad_r = 0, 0  
        elif self.kernel_size % 2 == 0:
            pad_l = (self.kernel_size - 2) // 2 + 1  
            pad_r = (self.kernel_size) // 2 + 1  
        else:
            pad_l = (self.kernel_size - 1) // 2 + 1 
            pad_r = (self.kernel_size - 1) // 2 + 1
        self.split = Splitting()

        modules_P = []
        modules_U = []
        modules_psi = []
        modules_phi = []
        prev_size = 1

        size_hidden = self.hidden_size
        modules_P += [
            nn.ReplicationPad1d((pad_l, pad_r)),
            nn.Conv1d(in_planes * prev_size, int(in_planes * size_hidden),
                      kernel_size=self.kernel_size, dilation=1, stride=1, groups= self.groups),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv1d(int(in_planes * size_hidden), in_planes, kernel_size=3, stride=1, groups= self.groups) if self.kernel_size != 1 else
                nn.Conv1d(int(in_planes * size_hidden), in_planes, kernel_size=self.kernel_size, stride=1, groups= self.groups),
            nn.Tanh()
        ]
        modules_U += [
            nn.ReplicationPad1d((pad_l, pad_r)),
            nn.Conv1d(in_planes * prev_size, int(in_planes * size_hidden),
                      kernel_size=self.kernel_size, dilation=1, stride=1, groups= self.groups),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv1d(int(in_planes * size_hidden), in_planes, kernel_size=3, stride=1, groups= self.groups) if self.kernel_size != 1 else
                nn.Conv1d(int(in_planes * size_hidden), in_planes, kernel_size=self.kernel_size, stride=1, groups= self.groups),
            nn.Tanh()
        ]
        modules_phi += [
            nn.ReplicationPad1d((pad_l, pad_r)),
            nn.Conv1d(in_planes * prev_size, int(in_planes * size_hidden),
                      kernel_size=self.kernel_size, dilation=1, stride=1, groups= self.groups),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv1d(int(in_planes * size_hidden), in_planes, kernel_size=3, stride=1, groups= self.groups) if self.kernel_size != 1 else
                nn.Conv1d(int(in_planes * size_hidden), in_planes, kernel_size=self.kernel_size, stride=1, groups= self.groups),
            nn.Tanh()
        ]
        modules_psi += [
            nn.ReplicationPad1d((pad_l, pad_r)),
            nn.Conv1d(in_planes * prev_size, int(in_planes * size_hidden),
                      kernel_size=self.kernel_size, dilation=1, stride=1, groups= self.groups),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv1d(int(in_planes * size_hidden), in_planes, kernel_size=3, stride=1, groups= self.groups) if self.kernel_size != 1 else
                nn.Conv1d(int(in_planes * size_hidden), in_planes, kernel_size=self.kernel_size, stride=1, groups= self.groups),
            nn.Tanh()
        ]
        self.phi = nn.Sequential(*modules_phi)
        self.psi = nn.Sequential(*modules_psi)
        self.P = nn.Sequential(*modules_P)
        self.U = nn.Sequential(*modules_U)

    def forward(self, x):
        (x_even, x_odd) = self.split(x)
        x_even = x_even.permute(0, 2, 1)
        x_odd = x_odd.permute(0, 2, 1)
        d = x_odd.mul(torch.exp(self.phi(x_even)))
        c = x_even.mul(torch.exp(self.psi(x_odd)))
        x_even_update = c + self.U(d)
        x_odd_update = d - self.P(c)
        return (x_even_update, x_odd_update)

class LevelSCINet(nn.Module):
    def __init__(self,in_planes, kernel_size, dropout, groups, hidden_size):
        super(LevelSCINet, self).__init__()
        self.interact = Interactor(in_planes= in_planes, kernel = kernel_size, dropout = dropout, groups = groups , hidden_size = hidden_size)

    def forward(self, x):
        (x_even_update, x_odd_update) = self.interact(x)
        return x_even_update.permute(0, 2, 1), x_odd_update.permute(0, 2, 1) #even: B, T, D odd: B, T, D

class SCINet_Tree(nn.Module):
    def __init__(self, in_planes, current_level, kernel_size, dropout, groups, hidden_size):
        super().__init__()
        self.current_level = current_level
        self.workingblock = LevelSCINet(
            in_planes = in_planes,
            kernel_size = kernel_size,
            dropout = dropout,
            groups= groups,
            hidden_size = hidden_size)
        if current_level!=0:
            self.SCINet_Tree_odd=SCINet_Tree(in_planes, current_level-1, kernel_size, dropout, groups, hidden_size)
            self.SCINet_Tree_even=SCINet_Tree(in_planes, current_level-1, kernel_size, dropout, groups, hidden_size)
    
    def zip_up_the_pants(self, even, odd):
        even = even.permute(1, 0, 2)
        odd = odd.permute(1, 0, 2) #L, B, D
        even_len = even.shape[0]
        odd_len = odd.shape[0]
        mlen = min((odd_len, even_len))
        _ = []
        for i in range(mlen):
            _.append(even[i].unsqueeze(0))
            _.append(odd[i].unsqueeze(0))
        if odd_len < even_len: 
            _.append(even[-1].unsqueeze(0))
        return torch.cat(_,0).permute(1,0,2) #B, L, D
        
    def forward(self, x):
        x_even_update, x_odd_update= self.workingblock(x)
        if self.current_level ==0:
            return self.zip_up_the_pants(x_even_update, x_odd_update)
        else:
            return self.zip_up_the_pants(self.SCINet_Tree_even(x_even_update), self.SCINet_Tree_odd(x_odd_update))

class SCINet(nn.Module):
    def __init__(self, output_len, input_len, input_dim = 7, hid_size = 1, num_levels = 3, groups = 1, kernel = 5, dropout = 0.5):
        super(SCINet, self).__init__()
        self.input_dim = input_dim
        self.input_len = input_len
        self.output_len = output_len
        self.hidden_size = hid_size
        self.num_levels = num_levels
        self.groups = groups
        self.kernel_size = kernel
        self.dropout = dropout

        self.blocks1 = SCINet_Tree(  
            in_planes = self.input_dim,
            current_level = self.num_levels - 1,
            kernel_size = self.kernel_size,
            dropout = self.dropout,
            groups = self.groups,
            hidden_size = self.hidden_size)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
        self.projection1 = nn.Conv1d(self.input_len, self.output_len, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        assert self.input_len % (np.power(2, self.num_levels)) == 0 
        res1 = x
        x = self.blocks1(x)
        x += res1
        x = self.projection1(x)
        return x

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.Linear = SCINet(output_len = self.pred_len, input_len = self.seq_len, input_dim = self.channels)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        x = self.Linear(x)
        return x # [Batch, Output length, Channel]