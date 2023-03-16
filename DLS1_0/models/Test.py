import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# class FFT_for_Topk(nn.Module):
#     def __init__(self, topk = 2):
#         super(FFT_for_Topk, self).__init__()
#         self.topk = topk
        
#     def forward(self, x):
#         xf = torch.fft.rfft(x, dim = 1)
#         frequency_list = abs(xf).mean(0).mean(-1)
#         frequency_list[0] = 0
#         values, _ = torch.topk(frequency_list, self.topk)
#         indices = frequency_list >= values[-1]
#         indices = indices.reshape(1, -1, 1)
#         xf_clean = indices * xf
#         x_clean = torch.fft.irfft(xf_clean, dim = 1)
#         return x_clean

# class series_decomp(nn.Module):
#     def __init__(self, topk):
#         super(series_decomp, self).__init__()
#         self.decomp = FFT_for_Topk(topk)
            
#     def forward(self, x):
#         x_clean = self.decomp(x)
#         x_res = x - x_clean
#         return x_res, x_clean
    
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.Linear = nn.Linear(self.seq_len, self.pred_len)
        
    def forward(self, x):
        # x: [Batch, Input length, Channel]
        x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x # [Batch, Output length, Channel]