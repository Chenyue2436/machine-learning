import torch
import torch.nn as nn
from models.DLSNet import DLSNet

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, output_channels, kernel_size, padding=0, kernels_per_layer=1):
        super(DepthwiseSeparableConv, self).__init__()
        # depthwise只考虑时间维度的交互， pointwise只考虑特征维度的交互
        self.depthwise = nn.Conv1d(in_channels, in_channels * kernels_per_layer, kernel_size=kernel_size, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv1d(in_channels * kernels_per_layer, output_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class DoubleConvDS(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, kernels_per_layer=1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            DepthwiseSeparableConv(in_channels, mid_channels, kernel_size=3, kernels_per_layer=kernels_per_layer, padding=1),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(mid_channels, out_channels, kernel_size=3, kernels_per_layer=kernels_per_layer, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class DownDS(nn.Module):
    def __init__(self, in_channels, out_channels, kernels_per_layer=1):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConvDS(in_channels, out_channels, kernels_per_layer=kernels_per_layer)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class UpDS(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len, pred_len, kernel_size = 24, kernels_per_layer=1):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        self.DLS = DLSNet(output_len = pred_len, input_len = seq_len, input_dim = in_channels // 2, kernel_size = kernel_size)
        self.conv = DoubleConvDS(in_channels, out_channels, in_channels // 2, kernels_per_layer=kernels_per_layer)
        

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x2 = self.DLS(x2.permute(0, 2, 1)).permute(0, 2, 1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, seq_len, pred_len, kernel_size = 24):
        super(UNet, self).__init__()
        # n_channels: in_feature_dim
        self.n_channels = n_channels
        self.down1 = DownDS(self.n_channels, self.n_channels, kernels_per_layer = 2)
        self.down2 = DownDS(self.n_channels, self.n_channels, kernels_per_layer = 2)
        self.DLS = DLSNet(output_len = pred_len // 4, input_len = seq_len // 4, input_dim = self.n_channels, kernel_size = kernel_size // 4)
        self.up1 = UpDS(2 * self.n_channels, self.n_channels, seq_len // 2, pred_len // 2, kernel_size = kernel_size // 2, kernels_per_layer = 2)
        self.up2 = UpDS(2 * self.n_channels, self.n_channels, seq_len, pred_len, kernel_size = kernel_size, kernels_per_layer = 2)

    def forward(self, x):
        x1 = x # [batch_size, feature_dim, input_length]
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x3 = self.DLS(x3.permute(0, 2, 1)).permute(0, 2, 1) # [batch_size, feature_dim, out_length]
        x = self.up1(x3, x2)
        x = self.up2(x, x1) # [batch_size, feature_dim, out_length]
        return x

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.feature_dim = configs.enc_in
        self.kernel_size = configs.moving_avg
        self.Linear1 = UNet(self.feature_dim, self.seq_len, self.pred_len, self.kernel_size)
        self.Linear2 = nn.Linear(self.pred_len, self.pred_len)
        
    def forward(self, x):
        # x: [Batch, Input length, Channel]
        x = x.permute(0, 2, 1)
        x = self.Linear1(x)
        x = self.Linear2(x)
        return x.permute(0, 2, 1) # [Batch, Output length, Channel]