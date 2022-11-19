import torch
import torch.nn as nn
from .base_model import BaseModel, as_np
from .focal_loss import BinaryFocalLoss
from .unet import UNet

class _unet_image2label(BaseModel):
    def __init__(self, opt, is_train=True):
        BaseModel.__init__(self, opt)
        self.opt = opt

        unet = UNet(1, 1, 16, True)
        self.netG = self.to_device(unet)

        if is_train:
            if opt['loss_type'] == 'focal':
                self.criterion = BinaryFocalLoss(0.75, 2, pos_weight=torch.tensor([10]).float().cuda(), is_logits=False)
            if opt['loss_type'] == 'BCE': 
                self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10]).float().cuda())
            if opt['loss_type'] == 'L1':
                self.criterion = nn.L1Loss()

            G_params = self.netG.parameters()

            lr = opt['lr']
            self.optimizer_G = torch.optim.Adam(G_params, lr=lr, amsgrad=True)
            # self.optimizer_D = torch.optim.Adam(D_params, lr=lr)

            self.register_buffer('real_label', torch.tensor(1.0).to(self.device))
            self.register_buffer('fake_label', torch.tensor(0.0).to(self.device))

            self.model_names = ['G']

        self.loss_G = 0
        # self.loss_D = 0
        self.loss_names = ['recon', 'GAN']

    def get_results(self):
        real_I = as_np(self.real_I)
        real_L = as_np(self.real_L)
        fake_L = as_np(self.fake_L)
        return {'real_I': real_I, 'fake_L': fake_L, 'real_L': real_L}

    def set_input(self, input):
        """
        input: 
            A: label
            B: collagen
        """
        self.index = input["index"]
        self.real_I = input['I'].to(self.device)
        self.real_L = input['L'].to(self.device)

    def forward(self, x):
        logits = self.netG(x)
        # x_recon = torch.sigmoid(logits)
        x_recon = logits
        return logits, x_recon
    
    def backward_G(self):
        self.loss_GAN = 0
        self.loss_recon = self.criterion(self.logits, self.real_L)
        self.loss_G = self.loss_recon
        
        self.loss_G.backward()

    def optimize_parameters(self):
        self.logits, self.fake_L = self.forward(self.real_I)

        # update G
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
    