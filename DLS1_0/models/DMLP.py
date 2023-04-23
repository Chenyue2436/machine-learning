import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.RevIN import RevIN
from statsmodels.tsa.seasonal import STL
import multiprocessing

class Splitting(nn.Module):
    def __init__(self, s):
        super(Splitting, self).__init__()
        self.s = s

    def forward(self, x):
        ans = []
        for i in range(self.s):
            ans.append(x[:, :, i::self.s].unsqueeze(-1))
        return torch.cat(ans, dim = -1).permute(0, 1, 3, 2)

class Catting(nn.Module):
    def __init__(self, s):
        super(Catting, self).__init__()
        self.s = s

    def forward(self, x):
        ans = []
        n = x.shape[-1]
        for i in range(n):
            for j in range(self.s):
                ans.append(x[:, :, j:j + 1, i:i + 1])
        return torch.cat(ans, dim = -1).permute(0, 1, 3, 2).squeeze(-1)
    
class MLP(nn.Module):
    def __init__(self, seq_len, s):
        super(MLP, self).__init__()
        self.seq_len = seq_len
        self.s = s
        
        self.split = Splitting(self.s)
        self.Linear1 = nn.Sequential(nn.Linear(self.seq_len // self.s, 2048 // self.s), nn.GELU(), nn.Linear(2048 // self.s, self.seq_len // self.s))
        self.Linear2 = nn.Sequential(nn.Linear(self.seq_len // self.s, 2048 // self.s), nn.GELU(), nn.Linear(2048 // self.s, self.seq_len // self.s))
        self.cat = Catting(self.s)
        
    def forward(self, x):# [Batch, Channel, Input length]
        x = self.split(x)
        x = self.Linear1(x)
        x = self.Linear2(x)
        x = self.cat(x)
        return x # [Batch, Channel, Input length]
    
class Decomp(nn.Module):
    def __init__(self, period):
        super(Decomp, self).__init__()
        self.kernel_size = period
        self.avg = nn.AvgPool1d(kernel_size = self.kernel_size, stride = 1, padding = 0)
        
    def forward(self, x):# [Batch, Input length, Channel]
        front = x[:, 0:1, :].repeat(1, self.kernel_size - 1 - (self.kernel_size - 1)//2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1)//2, 1)
        trend = torch.cat([front, x, end], dim=1)
        trend = self.avg(trend.permute(0, 2, 1))
        trend = trend.permute(0, 2, 1)
        seasonal = x - trend
        return trend, seasonal# [Batch, Input length, Channel]
        
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.feature_dim = configs.enc_in
        self.s = configs.s
        self.revin = configs.revin
        self.period = configs.moving_avg
        
        self.decomp = Decomp(self.period)
        self.revin_layer = RevIN(self.feature_dim)
        self.Linear_trend = nn.Sequential(nn.Linear(self.seq_len, self.pred_len))
        self.Linear_seasonal = nn.Sequential(MLP(self.seq_len, self.s), nn.Linear(self.seq_len, self.pred_len))
        
    def forward(self, x):# [Batch, Input length, Channel]
        if self.revin:
            x = self.revin_layer(x, 'norm')
        
        trend, seasonal = self.decomp(x)
        trend, seasonal = trend.permute(0, 2, 1), seasonal.permute(0, 2, 1)
        trend, seasonal = self.Linear_trend(trend), self.Linear_seasonal(seasonal)
        x = trend + seasonal
        x = x.permute(0, 2, 1)
        
        if self.revin:
            x = self.revin_layer(x, 'denorm')
        return x # [Batch, Output length, Channel]