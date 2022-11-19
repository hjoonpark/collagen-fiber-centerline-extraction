import os
import numpy as np
import itertools

import torch
import torch.nn as nn
from torch.nn import functional as F
from modules.utils.util_model import get_available_devices
import functools

class UNet(nn.Module):
    def __init__(self, params, is_train, logger, img_channel=1):
        super().__init__()
        self.device, self.gpu_ids = get_available_devices(logger)

        # define model
        self.unet = UNetModule(input_nc=img_channel, output_nc=img_channel, n_filters=params["unet"]["n_filters"], bilinear=True).to(self.device)

        if is_train:
            lr = params["train"]["lr"]
            loss_type = params["unet"]["loss_type"]
            if loss_type == 'focal':
                self.criterion = BinaryFocalLoss(0.75, 2, pos_weight=torch.tensor([10]).float().cuda(), is_logits=False)
            elif loss_type == 'BCE': 
                self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10]).float().cuda())
            elif loss_type == 'L1':
                self.criterion = nn.L1Loss()
            else:
                raise NotImplementedError(f"[{self.__class__.__name__ }] invalid loss type.")

            unet_params = self.unet.parameters()

            self.optimizer = torch.optim.Adam(unet_params, lr=lr, amsgrad=True)

            self.register_buffer('real_label', torch.tensor(1.0).to(self.device))
            self.register_buffer('fake_label', torch.tensor(0.0).to(self.device))

            self.model_names = ['unet']

        self.loss_recon = 0
        self.loss_names = ['recon']

    def set_input(self, data):
        self.filename = data["filename"]
        self.image = data["image"].to(self.device) # (B, img_channel, 256, 256)
        self.centerline = data["centerline"].to(self.device) # (B, img_channel, 256, 256)

    def optimize_parameters(self, iters, epoch):
        self.centerline_recon = self.forward(self.image)

        # update UNet
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()

    def forward(self, image):
        centerline_recon = self.unet(image)
        return centerline_recon

    def backward(self):
        self.loss_recon = self.criterion(self.centerline_recon, self.centerline)
        self.loss_recon.backward()

class UNetModule(nn.Module):
    def __init__(self, input_nc, output_nc, n_filters=64, bilinear=False):
        super().__init__()
        self.input_nc = input_nc
        self.bilinear = bilinear

        self.inc = DoubleConv(input_nc, n_filters)
        self.down1 = Down(n_filters, n_filters*2)
        self.down2 = Down(n_filters*2, n_filters*4)
        self.down3 = Down(n_filters*4, n_filters*8)
        factor = 2 if bilinear else 1
        self.down4 = Down(n_filters*8, n_filters*16 // factor)
        self.up1 = Up(n_filters*16, n_filters*8 // factor, bilinear)
        self.up2 = Up(n_filters*8, n_filters*4 // factor, bilinear)
        self.up3 = Up(n_filters*4, n_filters*2 // factor, bilinear)
        self.up4 = Up(n_filters*2, n_filters, bilinear)
        self.outc = OutConv(n_filters, output_nc)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return torch.sigmoid(logits)
        
""" Parts of the U-Net model """

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

def binary_focal_loss(input: torch.Tensor, target: torch.Tensor, alpha=0.25, gamma=2, reduction='mean', pos_weight=None, is_logits=False) -> torch.Tensor:
    
    if not pos_weight:
        pos_weight = torch.ones(input.shape[1], device=input.device, dtype=input.dtype)
    
    if not is_logits:
        p = input.sigmoid()
    else:
        p = input
        
    ce_loss = F.binary_cross_entropy_with_logits(input, target, reduction="none", pos_weight=pos_weight)
    
    p_t = p * target + (1 - p) * (1 - target)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * target + (1 - alpha) * (1 - target)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss
        
class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean', pos_weight=None, is_logits=False):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.pos_weight = pos_weight
        self.is_logits = is_logits
        
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return binary_focal_loss(input, target, self.alpha, self.gamma, self.reduction, self.pos_weight, self.is_logits)