import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.fft import rfft,rfftfreq
from torch.fft import irfft
from utils.RevIN import RevIN

class polynomial_interpolation(nn.Module):
    def __init__(self, kernel_size, seq_len, order = 4):
        super(polynomial_interpolation, self).__init__()
        self.kernel_size = kernel_size if seq_len % kernel_size == 0 else 24
        self.seq_len = seq_len
        self.matrix = []
        for i in range(1, self.kernel_size + 1):
            self.matrix.append([pow(i, j) for j in range(order)])
        self.matrix = torch.tensor(self.matrix).to(dtype = torch.float32)
        self.matrix = self.matrix @ ((self.matrix.t() @ self.matrix).inverse()) @ self.matrix.t()
    
    def forward(self, x):
        ans= [self.matrix.to(x.device) @ x[:, i:i + self.kernel_size, :] for i in range(0, self.seq_len, self.kernel_size)]
        return torch.cat(ans, dim = 1)

class main_frequency(nn.Module):
    def __init__(self, k = 2):
        super(main_frequency, self).__init__()
        self.k = k
    
    def forward(self, x):
        xf = rfft(x, dim = 1)
        xf_abs = torch.abs(xf)
        values, _ = torch.topk(xf_abs, self.k, dim = 1)
        indices = xf_abs >= values[:, -1:, :]
        xf_clean = indices * xf
        seasonal = irfft(xf_clean, dim = 1)
        return seasonal
    
class Decomp(nn.Module):
    def __init__(self, kernel_size, seq_len, order = 4, k = 2):
        super(Decomp, self).__init__()
        self.polynomial_interpolation = polynomial_interpolation(kernel_size, seq_len, order)
        self.main_frequency = main_frequency(k)
        
    def forward(self, x):
        trend = self.polynomial_interpolation(x)
        seasonal = self.main_frequency(x - trend)
        res = x - trend - seasonal
        return trend, seasonal, res
    
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
        
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.kernel_size = configs.moving_avg
        
        self.decomp = Decomp(self.kernel_size, self.seq_len)
        self.Linear_trend = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_seasonal = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_res = nn.Linear(self.seq_len, self.pred_len)
        
    def forward(self, x):# [Batch, Input length, Channel]
        trend, seasonal, res = self.decomp(x)
        trend, seasonal, res = trend.permute(0, 2, 1), seasonal.permute(0, 2, 1), res.permute(0, 2, 1)
        x = self.Linear_trend(trend) + self.Linear_seasonal(seasonal) + self.Linear_res(res)
        x = x.permute(0, 2, 1)
        return x # [Batch, Output length, Channel]