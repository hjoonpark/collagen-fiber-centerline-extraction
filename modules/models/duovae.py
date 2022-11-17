import os
import numpy as np
import itertools

import torch
import torch.nn as nn
import torch.distributions as dist
from torch.nn import functional as F
from modules.utils.util_model import get_available_devices, View
from modules.utils.util_io import as_np

class DuoVAE(nn.Module):
    def __init__(self, params, is_train, logger, img_channel=1, y_dim=6):
        super().__init__()
        self.device, self.gpu_ids = get_available_devices(logger)
        self.img_channel = img_channel
        self.y_dim = y_dim
        
        # parameters
        z_dim = params["duovae"]["z_dim"]
        w_dim = params["duovae"]["w_dim"]
        hid_channel = params["duovae"]["hid_channel"]
        hid_dim_x = params["duovae"]["hid_dim_x"]
        hid_dim_y = params["duovae"]["hid_dim_y"]

        # define models
        self.encoder_x = EncoderX(img_channel, hid_channel, hid_dim_x, z_dim, w_dim).to(self.device)
        self.decoder_x = DecoderX(img_channel, hid_channel, hid_dim_x, z_dim, w_dim).to(self.device)
        self.encoder_y = EncoderY(y_dim, hid_dim_y, w_dim).to(self.device)
        self.decoder_y = DecoderY(y_dim, hid_dim_y, w_dim).to(self.device)

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
            pos_weight = torch.tensor([10]).float().to(self.device)
            self.criteriaBCEWithLogits = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='sum')

            # used by util.model to plot losses
            self.loss_names = ["x_recon", "y_recon", "y_recon2", "kl_div_z", "kl_div_w", "kl_div_w2"]

    def set_input(self, data):
        self.filename = data["filename"]
        self.x = data['centerline'].to(self.device) # (B, img_channel, 256, 256)
        self.y = data['property'].to(self.device) # (B, y_dim)

    def optimize_parameters(self, iters, epoch):
        # main VAE
        self.optimizer_x.zero_grad()
        self.forward_backward_x()
        self.optimizer_x.step()
        
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
        # * reconstruction losses are rescaled w.r.t. image and label dimensions so that hyperparameters are easier to tune and consistent regardless of their dimensions.
        batch_size, _, h, w = self.x.shape
        
        self.loss_x_recon = self.criteriaBCEWithLogits(x_logits, self.x) / (batch_size*h*w)
        self.loss_y_recon = F.mse_loss(self.y_recon, self.y, reduction="sum") / (batch_size*self.y_dim)

        Pz = dist.Normal(torch.zeros_like(self.z), torch.ones_like(self.z))
        self.loss_kl_div_z = self.kl_divergence(Qz, Pz) / batch_size

        with torch.no_grad(): # no backpropagation on the encoder q(y|w) during this step
            w_mean, w_logvar = self.encoder_y(self.y)
            w_std = torch.sqrt(torch.exp(w_logvar.detach()))
            Pw = dist.Normal(w_mean.detach(), w_std)
        self.loss_kl_div_w = self.kl_divergence(Qw, Pw) / batch_size

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
        self.loss_y_recon2 = F.mse_loss(self.y_recon2, self.y, reduction="sum") / batch_size

        Pw = dist.Normal(torch.zeros_like(self.w2), torch.ones_like(self.w2))
        self.loss_kl_div_w2 = self.kl_divergence(Qw2, Pw) / batch_size

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
        return dist.kl_divergence(Q, P).sum()
  
"""
Encoder q(z,w|x): Encode input x to latent variables (z, w)
"""  
class EncoderX(nn.Module):
    def __init__(self, img_channel, hid_channel, hid_dim, z_dim, w_dim):
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        
        flat_dim = np.product((128*img_channel, 16, 16))
        self.encoder = nn.Sequential(
            ResidualConv(img_channel, 16*img_channel, sampling="down"), # 128
            ResidualConv(16*img_channel, 32*img_channel, sampling="down"), # 64
            ResidualConv(32*img_channel, 64*img_channel, sampling="down"), # 32
            ResidualConv(64*img_channel, 128*img_channel, sampling="down"), # 16
            View((-1, flat_dim)),
            ResidualLinear(flat_dim, hid_dim),
            ResidualLinear(hid_dim, 2*(z_dim + w_dim)),
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

        fc_shape = (128*img_channel, 16, 16)
        self.decoder = nn.Sequential(
            ResidualLinear(z_dim + w_dim, hid_dim),
            ResidualLinear(hid_dim, np.product(fc_shape)),
            View((-1, fc_shape[0], fc_shape[1], fc_shape[2])),
            ResidualConv(128*img_channel, 64*img_channel, sampling="up"), # 32
            ResidualConv(64*img_channel, 32*img_channel, sampling="up"), # 64
            ResidualConv(32*img_channel, 16*img_channel, sampling="up"), # 128
            ResidualConv(16*img_channel, 8*img_channel, sampling="up"), # 256
            nn.Conv2d(8*img_channel, 1, kernel_size=3, stride=1, padding=1)
        )
        # self.decoder = nn.Sequential(
        #     nn.Linear(z_dim+w_dim, hid_dim),
        #     nn.LeakyReLU(0.2, True),
        #     nn.Linear(hid_dim, hid_dim),
        #     nn.LeakyReLU(0.2, True),
        #     nn.Linear(hid_dim, 64*img_channel*16*16),
        #     nn.LeakyReLU(0.2, True),
        #     View((-1, 64*img_channel, 16, 16)),
        #     nn.ConvTranspose2d(64*img_channel, 32*img_channel, kernel_size=4, stride=2, padding=1, bias=False), # (32, 32, 32)
        #     nn.BatchNorm2d(32*img_channel),
        #     nn.LeakyReLU(0.2, True),
        #     nn.ConvTranspose2d(32*img_channel, 16*img_channel, kernel_size=4, stride=2, padding=1, bias=False), # (32, 64, 64)
        #     nn.BatchNorm2d(16*img_channel),
        #     nn.LeakyReLU(0.2, True),
        #     nn.ConvTranspose2d(16*img_channel, 8*img_channel, kernel_size=4, stride=2, padding=1, bias=False), # (32, 128, 128)
        #     nn.BatchNorm2d(8*img_channel),
        #     nn.LeakyReLU(0.2, True),
        #     nn.ConvTranspose2d(8*img_channel, 4*img_channel, kernel_size=4, stride=2, padding=1, bias=False), # (32, 256, 256)
        #     nn.LeakyReLU(0.2, True),
        #     nn.Conv2d(4*img_channel, img_channel, kernel_size=3, stride=1, padding=1) # (img_channel, 64, 64)
        # )

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

# class DecoderX(nn.Module):
#     def __init__(self, img_channel, hid_channel, hid_dim, z_dim, w_dim):
#         super().__init__()

#         self.decoder0 = nn.Sequential(
#             nn.Linear(z_dim+w_dim, hid_dim),
#             nn.LeakyReLU(0.2, True),
#             nn.Linear(hid_dim, hid_dim),
#             nn.LeakyReLU(0.2, True),
#             nn.Linear(hid_dim, 64*img_channel*16*16),
#             nn.LeakyReLU(0.2, True),
#             View((-1, 64*img_channel, 16, 16)),
#         )
#         self.decoder1 = nn.Sequential(
#             nn.ConvTranspose2d(64*img_channel, 32*img_channel, kernel_size=4, stride=2, padding=1), # (32, 32, 32)
#             nn.BatchNorm2d(32*img_channel),
#             nn.LeakyReLU(0.2, True),
#         )
#         self.decoder2 = nn.Sequential(
#             nn.ConvTranspose2d(32*img_channel, 16*img_channel, kernel_size=4, stride=2, padding=1), # (32, 64, 64)
#             nn.BatchNorm2d(16*img_channel),
#             nn.LeakyReLU(0.2, True),
#         )
#         self.decoder3 = nn.Sequential(
#             nn.ConvTranspose2d(16*img_channel, 8*img_channel, kernel_size=4, stride=2, padding=1), # (32, 128, 128)
#             nn.BatchNorm2d(8*img_channel),
#             nn.LeakyReLU(0.2, True),
#         )
#         self.decoder4 = nn.Sequential(
#             nn.ConvTranspose2d(8*img_channel, 4*img_channel, kernel_size=4, stride=2, padding=1), # (32, 256, 256)
#             nn.BatchNorm2d(4*img_channel),
#             nn.LeakyReLU(0.2, True),
#         )
#         self.decoder5 = nn.Sequential(
#             nn.ConvTranspose2d(4*img_channel, img_channel, kernel_size=3, stride=1, padding=1) # (img_channel, 64, 64)
#         )

#         self.res1 = nn.ConvTranspose2d(64*img_channel, 32*img_channel, kernel_size=4, stride=2, padding=1)
#         self.res2 = nn.ConvTranspose2d(32*img_channel, 16*img_channel, kernel_size=4, stride=2, padding=1)
#         self.res3 = nn.ConvTranspose2d(16*img_channel, 8*img_channel, kernel_size=4, stride=2, padding=1)
#         self.res4 = nn.ConvTranspose2d(8*img_channel, 4*img_channel, kernel_size=4, stride=2, padding=1)

#     def forward(self, z, w):
#         zw = torch.cat([z, w],dim=-1)
#         h0 = self.decoder0(zw)

#         # decode x
#         h1 = self.decoder1(h0)
#         r1 = self.res1(h0)

#         h2 = self.decoder2(h1+r1)
#         r2 = self.res2(r1)

#         h3 = self.decoder3(h2+r2)
#         r3 = self.res3(r2)

#         h4 = self.decoder4(h3+r3)
#         r4 = self.res4(r3)

#         x_logits = self.decoder5(h4+r4)
#         x_recon = torch.sigmoid(x_logits)
#         return x_logits, x_recon

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