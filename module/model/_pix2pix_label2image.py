import os
import torch
import torch.nn as nn
import functools
from .base_model import BaseModel, as_np

class _pix2pix(BaseModel):
    def __init__(self, opt, is_train=True):
        BaseModel.__init__(self, opt)
        self.opt = opt

        self.netG = self.to_device(UnetGenerator(input_nc=1, output_nc=1, num_downs=2, ngf=32, norm_layer=nn.BatchNorm2d, use_dropout=is_train))

        if is_train:
            self.netD = self.define_D()

            self.lambda_recon = opt['lambda_recon']
            self.D_update_interval = opt['D_update_interval']
            self.criterionGAN = nn.MSELoss()
            self.criterionPixel = nn.BCELoss()

            G_params = self.netG.parameters()
            D_params = self.netD.parameters()

            lr = opt['lr']
            self.optimizer_G = torch.optim.Adam(G_params, lr=lr)
            self.optimizer_D = torch.optim.Adam(D_params, lr=lr)

            self.register_buffer('real_label', torch.tensor(1.0))
            self.register_buffer('fake_label', torch.tensor(0.0))

            self.model_names = ['D', 'G']
        else:
            self.model_names = ['G']

        self.loss_G = 0
        self.loss_D = 0
        self.loss_names = ['recon', 'GAN', 'D']

    def get_results(self):
        real_L = as_np(self.real_L)
        real_I = as_np(self.real_I)
        fake_I = as_np(self.fake_I)
        return {'real_L': real_L, 'fake_I': fake_I, 'real_I': real_I}

    def set_input(self, input):
        """
        input: 
            L: label
            I: collagen
        """
        self.index = input["index"]
        self.real_L = input['L'].to(self.device)
        self.real_I = input['I'].to(self.device)
    def forward(self, x):
        return self.netG(x)

    def backward_D(self):
        real_LB = torch.cat((self.real_L, self.real_I), 1)
        fake_AB = torch.cat((self.real_L, self.fake_I), 1)
        pred_real = self.netD(real_LB)
        pred_fake = self.netD(fake_AB.detach())

        is_real = self.real_label.expand_as(pred_real).to(self.device)
        is_fake = self.fake_label.expand_as(pred_fake).to(self.device)

        loss_D_real = self.criterionGAN(pred_real, is_real)
        loss_D_fake = self.criterionGAN(pred_fake, is_fake)

        self.loss_D = (loss_D_real + loss_D_fake) * 0.5
        self.loss_D.backward()
    
    def backward_G(self):
        fake_AB = torch.cat((self.real_L, self.fake_I), 1)
        pred_fake = self.netD(fake_AB)
        is_real = self.real_label.expand_as(pred_fake).to(self.device)
        self.loss_GAN = self.criterionGAN(pred_fake, is_real)

        self.loss_recon = self.criterionPixel(self.fake_I, self.real_I)

        self.loss_G = self.lambda_recon*self.loss_recon + self.loss_GAN
        self.loss_G.backward()

    def optimize_parameters(self, iters):
        self.fake_I = self.forward(self.real_L)

        # update D
        if iters % self.D_update_interval == 0:
            self.set_requires_grad(self.netD, True)
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def define_D(self):
        """
        'PatchGAN' classifier described in the original pix2pix paper.
        """
        seq = [
            nn.Conv2d(2, 32, kernel_size=4, stride=2, padding=1, bias=True), # 128
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False), # 64
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False), # 32
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=1, bias=False), # 16
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=1, bias=True)
        ]
        return self.to_device(nn.Sequential(*seq))

class UnetGenerator(nn.Module):
    # https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """Create a Unet-based generator"""
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5): # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer
        

    def forward(self, input):
        """Standard forward"""
        output = self.model(input)
        return output

class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Sigmoid()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up
        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)