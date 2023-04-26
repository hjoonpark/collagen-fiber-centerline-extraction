import os
import numpy as np
import itertools

import torch
import torch.nn as nn
from torch.nn import functional as F
import functools

from modules.models.utils import *

class UNet(nn.Module):
    def __init__(self, params, is_train, img_channel=1, device='cpu'):
        super().__init__()
        self.device = device
        
        # define model
        self.unet = UNetModule(input_nc=img_channel, output_nc=img_channel, n_filters=params["unet"]["n_filters"], bilinear=True).to(self.device)

        if is_train:
            lr = params["train"]["lr"]
            loss_type = params["unet"]["loss_type"]
            if loss_type == 'focal':
                self.criterion = BinaryFocalLoss(0.75, 2, pos_weight=torch.tensor([10]).float().to(device), is_logits=False)
            elif loss_type == 'BCE': 
                self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10]).float().to(device))
            elif loss_type == 'L1':
                self.criterion = nn.L1Loss()
            else:
                raise NotImplementedError(f"[{self.__class__.__name__ }] invalid loss type.")

            unet_params = self.unet.parameters()

            self.optimizer = torch.optim.Adam(unet_params, lr=lr, amsgrad=True)

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
