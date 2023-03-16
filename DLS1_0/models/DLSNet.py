import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class moving_avg(nn.Module):
    def __init__(self, kernel_size):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, self.kernel_size - 1 - (self.kernel_size - 1)//2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1)//2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.decomp = moving_avg(kernel_size)
            
    def forward(self, x):
        trend = self.decomp(x)
        res = x - trend
        return res, trend

class Splitting(nn.Module):
    def __init__(self, kernel_size):
        super(Splitting, self).__init__()
        self.decompsition = series_decomp(kernel_size)

    def forward(self, x):
        # x.shape: [Batch_size, input_length, feature_dim]
        season, trend = self.decompsition(x)
        return (season, trend)

class Interactor(nn.Module):
    def __init__(self, in_planes, kernel_size, kernel = 5, dropout=0.5, groups = 1, hidden_size = 1):
        super(Interactor, self).__init__()
        self.kernel_size = kernel
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.groups = groups
        if self.kernel_size == 1:
            pad_l, pad_r = 0, 0  
        elif self.kernel_size % 2 == 0:
            pad_l = (self.kernel_size - 2) // 2 + 1 #by default: stride==1 
            pad_r = (self.kernel_size) // 2 + 1 #by default: stride==1 

        else:
            pad_l = (self.kernel_size - 1) // 2 + 1 # we fix the kernel size of the second layer as 3.
            pad_r = (self.kernel_size - 1) // 2 + 1
        self.split = Splitting(kernel_size)

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
        (x_season, x_trend) = self.split(x)
        x_season = x_season.permute(0, 2, 1) # [batch_size, featured_dim, input_length]
        x_trend = x_trend.permute(0, 2, 1)
        d = x_trend.mul(torch.exp(self.phi(x_season)))
        c = x_season.mul(torch.exp(self.psi(x_trend)))
        x_season_update = c + self.U(d)
        x_trend_update = d - self.P(c)
        return (x_season_update, x_trend_update) # [batch_size, featured_dim, input_length]

class LevelSCINet(nn.Module):
    def __init__(self, in_planes, kernel_size, kernel, dropout, groups, hidden_size):
        super(LevelSCINet, self).__init__()
        self.interact = Interactor(in_planes = in_planes, kernel_size = kernel_size, kernel = kernel, dropout = dropout, groups = groups , hidden_size = hidden_size)

    def forward(self, x):
        (x_season_update, x_trend_update) = self.interact(x)
        return x_season_update.permute(0, 2, 1), x_trend_update.permute(0, 2, 1) # [batch_size, input_length, featured_dim]

class DLSNet_Tree(nn.Module):
    def __init__(self, in_planes, in_levels, kernel_size, seq_len, kernel, dropout, groups, hidden_size):
        super().__init__()
        self.in_levels = in_levels
        self.Linear_Interact = nn.ModuleList()
        self.Linear_Trend = nn.ModuleList()
        for i in range(self.in_levels):
            self.Linear_Interact.append(LevelSCINet(in_planes = in_planes,
                            kernel_size = kernel_size, 
                            kernel = kernel,
                            dropout = dropout,
                            groups= groups,
                            hidden_size = hidden_size))
            self.Linear_Trend.append(nn.Linear(seq_len, seq_len))
        self.norm_layer = nn.BatchNorm1d(in_planes)
        
    def forward(self, x):
        # x: [Batch, Input length, Channel]  
        trend_output =  []
        seasonal_output = x
        for i in range(self.in_levels):
            seasonal_init, trend_init = self.Linear_Interact[i](seasonal_output)
            trend_init = trend_init.permute(0, 2, 1)
            trend_output.append(self.Linear_Trend[i](trend_init).permute(0, 2, 1))
            seasonal_output = seasonal_init
        x = self.norm_layer((sum(trend_output) + seasonal_output).permute(0, 2, 1)).permute(0, 2, 1)
        return x # [Batch, Output length, Channel]

class DLSNet(nn.Module):
    def __init__(self, output_len, input_len, input_dim, kernel_size, hid_size = 1, num_levels = 3, groups = 1, kernel = 5, dropout = 0.5):
        super(DLSNet, self).__init__()
        self.input_dim = input_dim
        self.input_len = input_len
        self.output_len = output_len
        self.kernel_size = kernel_size
        self.hidden_size = hid_size
        self.num_levels = num_levels
        self.groups = groups
        self.kernel = kernel
        self.dropout = dropout
        self.blocks1 = DLSNet_Tree(  
            in_planes = self.input_dim,
            in_levels = self.num_levels,
            kernel_size = self.kernel_size,
            seq_len = self.input_len,
            kernel = self.kernel,
            dropout = self.dropout,
            groups = self.groups,
            hidden_size = self.hidden_size)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel[0] * m.kernel[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
        self.projection1 = nn.Conv1d(self.input_len, self.output_len, kernel_size=1, stride=1, bias=False)
        
    def forward(self, x):
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
        self.feature_dim = configs.enc_in
        self.in_levels = 3
        kernel_size = configs.moving_avg  
        self.Linear = DLSNet(output_len = self.pred_len, input_len = self.seq_len, input_dim = self.feature_dim, kernel_size = kernel_size, num_levels = self.in_levels)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        x = self.Linear(x)
        return x # [Batch, Output length, Channel]