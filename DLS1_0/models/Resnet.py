import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.DLSNet import DLSNet
    
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        kernel_size = configs.moving_avg
        self.Linear = nn.ModuleList()
        for i in range(3):
            self.Linear.append(DLSNet(output_len = self.pred_len // np.power(2, i), input_len = self.seq_len, 
                             input_dim = self.channels, kernel_size = kernel_size))
        self.upsample = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        y = []
        for cur_level in range(3):
            temp = self.Linear[cur_level](x).permute(0, 2, 1)
            for _ in range(cur_level):
                temp = self.upsample(temp)
            y.append(temp.permute(0, 2, 1))
        return sum(y) # [Batch, Output length, Channel]