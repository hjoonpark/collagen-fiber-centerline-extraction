import os
import numpy as np
import itertools

import torch
import torch.nn as nn
import torch.distributions as dist
from torch.nn import functional as F
from modules.utils.util_model import View
from modules.utils.util_io import as_np
from modules.models.utils import *

class DuoVAE(nn.Module):
    def __init__(self, params, is_train, img_channel=1, y_dim=6, device='cpu'):
        super().__init__()
        self.img_channel = img_channel
        self.y_dim = y_dim
        self.device = device
        
        # parameters
        z_dim = params["duovae"]["z_dim"]
        w_dim = params["duovae"]["w_dim"]
        hid_channel = params["duovae"]["hid_channel"]
        hid_dim_x = params["duovae"]["hid_dim_x"]
        hid_dim_y = params["duovae"]["hid_dim_y"]

        # define models
        self.encoder_x = EncoderX(img_channel, hid_channel, hid_dim_x, z_dim, w_dim)
        self.decoder_x = DecoderX(img_channel, hid_channel, hid_dim_x, z_dim, w_dim)
        self.encoder_y = EncoderY(y_dim, hid_dim_y, w_dim)
        self.decoder_y = DecoderY(y_dim, hid_dim_y, w_dim)

        # used by util.model to save/load model
        self.model_names = ["encoder_x", "decoder_x", "encoder_y", "decoder_y"]

        if is_train:
            # parameters used for training
            lr = params["train"]["lr"]
            self.x_recon_weight = params["duovae"]["x_recon_weight"]
            self.y_recon_weight = params["duovae"]["y_recon_weight"]
            self.beta_z = params["duovae"]["beta_z"]
            self.beta_w = params["duovae"]["beta_w"]
            self.beta_w2 = params["duovae"]["beta_w2"]

            # optimizers
            params_x = itertools.chain(self.encoder_x.parameters(), self.decoder_x.parameters(), self.decoder_y.parameters())
            params_y = itertools.chain(self.encoder_y.parameters(), self.decoder_y.parameters())
            self.optimizer_x = torch.optim.Adam(params_x, lr=lr)
            self.optimizer_y = torch.optim.Adam(params_y, lr=lr)

            # losses
            loss_type = params["duovae"]["loss_type"]
            if loss_type == "focal":
                pos_weight = torch.tensor([10]).float().to(self.device)
                self.criterion = BinaryFocalLoss(0.75, 2, pos_weight=pos_weight, is_logits=False)
            elif loss_type == "BCE":
                pos_weight = torch.tensor([10]).float()
                self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='mean')

            # used by util.model to plot losses
            self.loss_names = ["x_recon", "y_recon", "y_recon2", "kl_div_z", "kl_div_w", "kl_div_w2"]

    def set_input(self, data):
        self.filename = data["filename"]
        self.x = data['centerline'].to(self.device) # (B, img_channel, 256, 256)
        self.y = data['property'].to(self.device) # (B, y_dim)

    def optimize_parameters(self, iters, epoch):
        # main VAE
        self.encoder_y.requires_grad = False
        self.optimizer_x.zero_grad()
        self.forward_backward_x()
        self.optimizer_x.step()
        self.encoder_y.requires_grad = True
        
        # auxiliary VAE
        self.optimizer_y.zero_grad()
        self.forward_backward_y()
        self.optimizer_y.step()

    def encode(self, x, sample: bool):
        # alias for encode_x
        return self.encode_x(x, sample)

    def encode_x(self, x, sample: bool):
        # encode
        (z_mean, w_mean), (z_logvar, w_logvar) = self.encoder_x(x)
        # sample w, z
        z, Qz = self.reparameterize(z_mean, z_logvar, sample)
        w, Qw = self.reparameterize(w_mean, w_logvar, sample)
        return (z, w), (Qz, Qw)

    def decode(self, z, w):
        # alias for decode_x
        return self.decode_x(z, w)
                
    def decode_x(self, z, w):
        y_recon = self.decoder_y(w)
        x_logits, x_recon = self.decoder_x(z, w)
        return x_logits, x_recon, y_recon

    def encode_y(self, y, sample: bool):
        # encode
        w_mean, w_logvar = self.encoder_y(y)
        # sample w
        w2, Qw2 = self.reparameterize(w_mean, w_logvar, sample=sample)
        return w2, Qw2

    def decode_y(self, w2):
        return self.decoder_y(w2)

    def forward_backward_x(self):
        # encode
        (self.z, self.w), (Qz, Qw) = self.encode_x(self.x, sample=True)
        # decode
        x_logits, self.x_recon, self.y_recon = self.decode_x(self.z, self.w)

        # losses
        batch_size, _, h, w = self.x.shape
        
        self.loss_x_recon = self.criterion(x_logits, self.x)
        self.loss_y_recon = F.mse_loss(self.y_recon, self.y, reduction="mean")

        Pz = dist.Normal(torch.zeros_like(self.z), torch.ones_like(self.z))
        self.loss_kl_div_z = self.kl_divergence(Qz, Pz)

        w_mean, w_logvar = self.encoder_y(self.y)
        w_std = torch.exp(0.5*w_logvar)
        Pw = dist.Normal(w_mean.detach(), w_std.detach())
        self.loss_kl_div_w = self.kl_divergence(Qw, Pw)

        loss = self.x_recon_weight*self.loss_x_recon + self.y_recon_weight*self.loss_y_recon \
                + self.beta_z*self.loss_kl_div_z + self.beta_w*self.loss_kl_div_w
        loss.backward()

    def forward_backward_y(self):
        # encode
        self.w2, Qw2 = self.encode_y(self.y, sample=True)
        # decode
        self.y_recon2 = self.decoder_y(self.w2)

        # losses
        batch_size = self.x.shape[0]
        self.loss_y_recon2 = F.mse_loss(self.y_recon2, self.y, reduction="mean")

        Pw = dist.Normal(torch.zeros_like(self.w2), torch.ones_like(self.w2))
        self.loss_kl_div_w2 = self.kl_divergence(Qw2, Pw)

        loss = self.y_recon_weight*self.loss_y_recon2 + self.beta_w2*self.loss_kl_div_w2
        loss.backward()

    def traverse_y(self, x, y, y_mins, y_maxs, n_samples):
        x = x.to(self.device)
        y = y.to(self.device)

        unit_range = torch.arange(0, 1+1e-5, 1.0/(n_samples-1))

        (z, _), _ = self.encode(x[[0]], sample=False)

        _, n_channel, h, w = x.shape
        vdivider = np.ones((1, n_channel, h, 1))
        hdivider = np.ones((1, n_channel, 1, w*n_samples + (n_samples-1)))
        # traverse
        x_recons_all = None
        for y_idx in range(len(y_mins)):
            x_recons = None
            for a in unit_range:
                y_new = torch.clone(y[[0]]).cpu() # had to move to cpu for some internal bug in the next line (Apple silicon-related)
                y_new[0, y_idx] = y_mins[y_idx]*(1-a) + y_maxs[y_idx]*a
                y_new = y_new.to(self.device)

                # encode for w
                w, _ = self.encoder_y(y_new)

                # decode
                _, x_recon, _ = self.decode(z, w)
                x_recons = as_np(x_recon) if x_recons is None else np.concatenate((x_recons, vdivider, as_np(x_recon)), axis=-1)
            x_recons_all = x_recons if x_recons_all is None else np.concatenate((x_recons_all, hdivider, x_recons), axis=2)
        x_recons_all = np.transpose(x_recons_all, (0, 2, 3, 1))
        return x_recons_all
          
    def reparameterize(self, mean, logvar, sample):
        if sample:
            std = torch.exp(0.5*logvar)
            P = dist.Normal(mean, std)
            z = P.rsample()
            return z, P
        else:
            return mean, None

    def kl_divergence(self, Q, P):
        batch_size, z_dim = Q.loc.shape
        return dist.kl_divergence(Q, P).mean()
    
"""
Encoder q(z,w|x): Encode input x to latent variables (z, w)
"""  
class EncoderX(nn.Module):
    def __init__(self, img_channel, hid_channel, hid_dim, z_dim, w_dim):
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        
        # flat_dim = np.product((hid_channel, 16, 16))
        # self.encoder = nn.Sequential(
        #     ResidualConv(img_channel, 4, sampling="down"), # 128
        #     ResidualConv(4, 8, sampling="down"), # 64
        #     ResidualConv(8, 16, sampling="down"), # 32
        #     ResidualConv(16, 32, sampling="down"), # 32
        #     ResidualConv(32, 64, sampling="down"), # 16
        #     View((-1, 64*8*8)),
        #     nn.Linear(64*8*8, hid_dim),
        #     nn.LeakyReLU(0.2, True),
        #     nn.Linear(hid_dim, 2*(z_dim + w_dim)),
        # )

        # self.encoder = nn.Sequential(
        #     nn.Conv2d(img_channel, 8, kernel_size=3, stride=2, padding=1, bias=True), # (128, 128)
        #     nn.LeakyReLU(0.2, True),
        #     nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, bias=True), # (64, 64)
        #     nn.LeakyReLU(0.2, True),
        #     nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=True), # (32, 32)
        #     nn.LeakyReLU(0.2, True),
        #     nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=True), # (16, 16)
        #     nn.LeakyReLU(0.2, True),
        #     nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=True), # (8, 8)
        #     nn.LeakyReLU(0.2, True),
        #     View((-1, 128*8*8)),
        #     nn.Linear(128*8*8, hid_dim),
        #     nn.LeakyReLU(0.2, True),
        #     nn.Linear(hid_dim, 2*(z_dim + w_dim)),
        # )

        # self.encoder = nn.Sequential(
        #     ResBlockDownsample(1, 4, bn=True), # 128
        #     ResBlockDownsample(4, 8, bn=True), # 64
        #     ResBlockDownsample(8, 16, bn=True), # 32
        #     ResBlockDownsample(16, 32, bn=True), # 16
        #     ResBlockDownsample(32, 64, bn=True), # 8
        #     View((-1, 64*8*8)),
        #     nn.Linear(64*8*8, hid_dim),
        #     nn.LeakyReLU(0.2, True),
        #     nn.Linear(hid_dim, 2*(z_dim + w_dim))
        # )

        self.encoder = nn.Sequential(
            DoubleConv(img_channel, 16),
            Down(16, 16), # 128
            Down(16, 32), # 64
            Down(32, 32), # 32
            Down(32, 64), # 16
            Down(64, 64), # 8
            View((-1, 64*8*8)),
            nn.Linear(64*8*8, hid_dim),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hid_dim, 2*(z_dim + w_dim))
        )

    def forward(self, x):
        mean_logvar = self.encoder(x)
        mean_logvar_z = mean_logvar[:, 0:2*self.z_dim]
        mean_logvar_w = mean_logvar[:, 2*self.z_dim:]

        # z
        z_mean, z_logvar = mean_logvar_z.view(-1, self.z_dim, 2).unbind(-1)

        # w
        w_mean, w_logvar = mean_logvar_w.view(-1, self.w_dim, 2).unbind(-1)

        return (z_mean, w_mean), (z_logvar, w_logvar)

"""
Decoder p(x|z,w): Recontruct input x from latent variables (z, w)
"""  
class DecoderX(nn.Module):
    def __init__(self, img_channel, hid_channel, hid_dim, z_dim, w_dim):
        super().__init__()

        # fc_shape = (hid_channel, 16, 16)
        # self.decoder = nn.Sequential(
        #     nn.Linear(z_dim + w_dim, hid_dim),
        #     nn.LeakyReLU(0.2, True),
        #     nn.Linear(hid_dim, 64*8*8),
        #     nn.LeakyReLU(0.2, True),
        #     View((-1, 64, 8, 8)),
        #     ResidualConv(64, 32, sampling="up"), # 32
        #     ResidualConv(32, 16, sampling="up"), # 128
        #     ResidualConv(16, 8, sampling="up"), # 256
        #     ResidualConv(8, 4, sampling="up"), # 256
        #     ResidualConv(4, 2, sampling="up"), # 256
        #     nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1)
        # )

        # self.decoder = nn.Sequential(
        #     nn.Linear(z_dim+w_dim, hid_dim),
        #     nn.LeakyReLU(0.2, True),
        #     nn.Linear(hid_dim, 128*8*8),
        #     nn.LeakyReLU(0.2, True),
        #     View((-1, 128, 8, 8)),
        #     nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False), # (32, 16, 16)
        #     nn.BatchNorm2d(64),
        #     nn.LeakyReLU(0.2, True),
        #     nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False), # (32, 32, 32)
        #     nn.BatchNorm2d(32),
        #     nn.LeakyReLU(0.2, True),
        #     nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, bias=False), # (32, 64, 64)
        #     nn.BatchNorm2d(16),
        #     nn.LeakyReLU(0.2, True),
        #     nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1, bias=False), # (32, 128, 128)
        #     nn.BatchNorm2d(8),
        #     nn.LeakyReLU(0.2, True),
        #     # nn.ConvTranspose2d(8, img_channel, kernel_size=4, stride=2, padding=1, bias=True), # (32, 256, 256)
        # )

        # self.conv_last = nn.ConvTranspose2d(8, img_channel, kernel_size=4, stride=2, padding=1, bias=True) # (32, 256, 256)
        # self.conv_last.weight.data.fill_(0.0)
        # self.conv_last.bias.data.fill_(0.0)

        # self.decoder = nn.Sequential(
        #     nn.Linear(z_dim+w_dim, hid_dim),
        #     nn.LeakyReLU(0.2, True),
        #     nn.Linear(hid_dim, 64*8*8),
        #     nn.LeakyReLU(0.2, True),
        #     View((-1, 64, 8, 8)),
        #     ResBlockUpsample(64, 32, bn=True), # 8
        #     ResBlockUpsample(32, 16, bn=True), # 16
        #     ResBlockUpsample(16, 8, bn=True), # 64
        #     ResBlockUpsample(8, 4, bn=True), # 128
        #     ResBlockUpsample(4, 2, bn=True), # 256
        #     nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1, bias=True)
        # )

        self.decoder = nn.Sequential(
            nn.Linear(z_dim+w_dim, hid_dim),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hid_dim, 64*8*8),
            nn.LeakyReLU(0.2, True),
            View((-1, 64, 8, 8)),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            DoubleConv(64, 64),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            DoubleConv(64, 32),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            DoubleConv(32, 32),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            DoubleConv(32, 16),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            DoubleConv(16, 16),
            OutConv(16, img_channel)
        )

    def forward(self, z, w):
        zw = torch.cat([z, w],dim=-1)

        # decode x
        x_logits = self.decoder(zw)
        x_recon = torch.sigmoid(x_logits)
        return x_logits, x_recon

"""
Encoder q(w|y): Recontruct input y from latent variables w
"""  
class EncoderY(nn.Module):
    def __init__(self, y_dim, hid_dim, w_dim):
        super().__init__()
        self.w_dim = w_dim

        self.encoder = nn.Sequential(
            nn.Linear(y_dim, hid_dim),
            nn.ReLU(True),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(True),
            nn.Linear(hid_dim, w_dim*2),
        )

    def forward(self, y):
        mu_logvar = self.encoder(y)
        mu, logvar = mu_logvar.view(-1, self.w_dim, 2).unbind(-1)
        return mu, logvar

"""
Decoder p(y|w): Recontruct input y from latent variables w
"""  
class DecoderY(nn.Module):
    def __init__(self, y_dim, hid_dim, w_dim):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(w_dim, hid_dim),
            nn.ReLU(True),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(True),
            nn.Linear(hid_dim, y_dim),
        )
        
    def forward(self, w):
        y_recon = self.decoder(w)
        return y_recon

class ResidualLinear(nn.Module):
    def __init__(self, input_dim, latent_dim, dropout=0.0):
        super(ResidualLinear, self).__init__()

        seq = [
            nn.Linear(input_dim, latent_dim),
            nn.LeakyReLU(0.2, True)
        ]

        if dropout > 0.0:
            seq += [nn.Dropout(dropout)]
        # seq += [nn.Linear(latent_dim, latent_dim)]

        rseq = [nn.Linear(input_dim, latent_dim)]

        self.fc = nn.Sequential(*seq)
        self.rfc = nn.Sequential(*rseq)
        self.lrelu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        out = self.fc(x)
        res = self.rfc(x)
        out = self.lrelu(res+out)
        return out

class ResidualConv(nn.Module):
    def __init__(self, in_channels, out_channels, sampling, use_batch_norm=True, dropout=0.0):
        super(ResidualConv, self).__init__()
        seq = []
        rseq = []
        if sampling == "down":
            # downsample
            seq += [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=not use_batch_norm)]
            rseq += [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=not use_batch_norm)]
        elif sampling == "up":
            # upsample
            seq += [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=not use_batch_norm)]
            rseq += [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=not use_batch_norm)]
        else:
            # keep dimensions
            seq += [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=not use_batch_norm)]
            rseq += [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=not use_batch_norm)]

        if use_batch_norm:
            seq += [nn.BatchNorm2d(out_channels)]
        seq += [nn.LeakyReLU(0.2, True)]

        if dropout > 0.0:
            seq += [nn.Dropout(dropout)]
        
        # seq += [nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=not use_batch_norm)]
    
        if use_batch_norm:
            seq += [nn.BatchNorm2d(out_channels)]
            rseq += [nn.BatchNorm2d(out_channels)]
        self.net = nn.Sequential(*seq)
        self.rnet = nn.Sequential(*rseq)
        self.lrelu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        out = self.net(x)
        res = self.rnet(x)
        out = self.lrelu(res + out)
        return out

class ResBlockDownsample(nn.Module):
    def __init__(self, in_channels, out_channels, bn):
        super(ResBlockDownsample, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=not bn)
        self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=not bn)
        self.down_shortcut = nn.MaxPool2d(3, stride=2, padding=1)
        self.bn = bn
        if bn:
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        out = self.conv1(x)
        shortcut = self.conv_shortcut(x)
        shortcut = self.down_shortcut(shortcut)
        if self.bn:
            out = self.bn1(out)
            shortcut = self.bn2(shortcut)
        out += shortcut
        out = self.relu(out)
        return out

class ResBlockUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, bn):
        super(ResBlockUpsample, self).__init__()
        self.bn = bn
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=not bn)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=not bn)
        self.up_shortcut = nn.Upsample(scale_factor=2, mode='nearest')
        if bn:
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        out = self.conv1(x)
        shortcut = self.conv_shortcut(x)
        shortcut = self.up_shortcut(shortcut)
        if self.bn:
            out = self.bn1(out)
            shortcut = self.bn2(shortcut)
        out += shortcut
        out = self.relu(out)
        return out
