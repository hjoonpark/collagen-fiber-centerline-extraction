import os
import torch
import torch.nn as nn
from collections import OrderedDict
import random
import numpy as np
import argparse
import json

def as_np(x):
    return x.detach().cpu().numpy()
class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)

def load_opt(opt_path):
    with open(opt_path, 'r') as f:
        args = json.load(f)
        opt = argparse.ArgumentParser().parse_args()
        for k, v in args.items():
            setattr(opt, k, v)

    return opt

def setup_seed(seed):
    random.seed(seed)                          
    np.random.seed(seed)                       
    torch.manual_seed(seed)                    
    torch.cuda.manual_seed(seed)               
    torch.cuda.manual_seed_all(seed)           
    torch.backends.cudnn.deterministic = True  

class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x
class MLP(nn.Module):
    def __init__(self, input_dim, mlp_dim, output_dim, n_mlp_layer=3):
        super(MLP, self).__init__()

        seq = []
        seq += [
            nn.Linear(input_dim, mlp_dim),
            nn.ReLU(True)
            ]
        for _ in range(n_mlp_layer-2):
            seq += [
                    nn.Linear(mlp_dim, mlp_dim),
                    nn.ReLU(True)
                ]
                
        seq += [nn.Linear(mlp_dim, output_dim)]
        self.model = nn.Sequential(*seq)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))

class AdaptiveInstanceNorm2d(nn.Module):
    """
    https://github.com/hjoonpark/MUNIT/blob/a82e222bc359892bd0f522d7a0f1573f3ec4a485/networks.py#L256
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = torch.nn.functional.batch_norm(x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'

class AdaINResnetBlock(nn.Module):
    def __init__(self, dim):
        super(AdaINResnetBlock, self).__init__()
        self.dim = dim
        self.conv_block = self.build_conv_block()

    def build_conv_block(self):
        dim = self.dim
        conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=True),
            AdaptiveInstanceNorm2d(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=True),
            AdaptiveInstanceNorm2d(dim)
        )
        return conv_block

    def forward(self, x):
        out = x + self.conv_block(x)
        return out
class ResnetBlock0(nn.Module):
    def __init__(self, dim):
        super(ResnetBlock0, self).__init__()
        self.dim = dim
        self.conv_block = self.build_conv_block()

    def build_conv_block(self):
        dim = self.dim
        conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
            nn.BatchNorm2d(dim)
        )
        return conv_block

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ResnetBlock(nn.Module):
    def __init__(self, dim, pad_type, norm, activation, dropout=0.0):
        super(ResnetBlock, self).__init__()

        seq = []

        # -------------------------------------------------------------------------
        # add padding
        if pad_type == 'reflect':
            seq += [nn.ReflectionPad2d(1)]
        else:
            assert False, 'Please set Resnet pad_type'
        # conv2d
        seq += [nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=True)]
        # normalization
        if norm == 'bn':
            seq += [nn.BatchNorm2d(dim)]
        elif norm == 'in':
            seq += [nn.InstanceNorm2d(dim)]
        else:
            assert (norm == 'none'), 'Please set Resnet norm'
        # activation
        if activation == 'relu':
            seq += [nn.ReLU(True)]
        elif activation == 'lrelu':
            seq += [nn.LeakyReLU(True)]
        else:
            assert False, 'Please set Resnet activation'
        # dropout
        if dropout > 0:
            seq += [nn.Dropout(dropout)]
        
        # -------------------------------------------------------------------------
        # add padding
        if pad_type == 'reflect':
            seq += [nn.ReflectionPad2d(1)]
        else:
            assert False, 'Please set Resnet pad_type'
        # conv2d
        seq += [nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=True)]
        # normalization
        if norm == 'bn':
            seq += [nn.BatchNorm2d(dim)]
        elif norm == 'in':
            seq += [nn.InstanceNorm2d(dim)]
        else:
            assert (norm == 'none'), 'Please set Resnet norm'
        self.conv_block = nn.Sequential(*seq)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class BaseModel(nn.Module):
    def __init__(self, opt):
        super(BaseModel, self).__init__()
        self.opt = opt
        self.loss_names = []
        self.model_names = []
        self.optimizers = []

        self.gpu_ids = self._load_device()
        self.device = torch.device('cuda:0' if (torch.cuda.is_available() and len(self.gpu_ids) > 0) else torch.device('cpu'))  # get device name: CPU or GPU
            
        if torch.cuda.device_count() > 0:
            print(torch.cuda.device_count(), "GPUs found. gpu_ids =", self.gpu_ids, ', device =', self.device)

    def to_device(self, net):
        if len(self.gpu_ids) == 1:
            net.to(self.device)
        elif len(self.gpu_ids) > 1:
            net.to(self.gpu_ids[0])
            net = torch.nn.DataParallel(net, self.gpu_ids)  # multi-GPUs
        return net

    def _load_device(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            n_gpu = torch.cuda.device_count()
            return [i for i in range(n_gpu)]
        else:
            return []

    def update_learning_rate(self):
        """
        called at the end of every epoch
        """
        old_lr = self.optimizers[0].param_groups[0]['lr']
        for scheduler in self.schedulers:
            if self.opt['lr_policy'] == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        return old_lr, lr

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret
    
    def get_nets(self):
        nets = {}
        for name in self.model_names:
            if isinstance(name, str):
                net_name = "net{}".format(name)
                net = getattr(self, net_name)
                nets[name] = net
        return nets

    def save_networks(self, save_dir):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                net_name = "net{}".format(name)
                net = getattr(self, net_name)
                save_path = os.path.join(save_dir, "{}.pt".format(net_name))
                
                if len(self.gpu_ids) == 1:
                    torch.save(net.cpu().state_dict(), save_path)
                    net.to(self.device)
                elif len(self.gpu_ids) > 1:
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.to(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def load_networks(self, model_dir, logger=None):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                net_name = "net{}".format(name)
                load_path = os.path.join(model_dir, "{}.pt".format(net_name))
                if not os.path.exists(load_path):
                    if logger is None:
                        print('\t>> [!] not found. skipping {}'.format(load_path))
                    else:
                        logger.write('\t>> [!] not found. skipping {}'.format(load_path))
                    continue

                if logger is None:
                    print('\t:: loading {} from {}'.format(net_name, load_path), end='')
                else:
                    logger.write('\t:: loading {} from {}'.format(net_name, load_path))
                net = getattr(self, net_name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device))
                if logger is None:
                    print(" {} states".format(len(state_dict.keys())))
                else:
                    logger.write("\t    {} states".format(len(state_dict.keys())))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                # for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                #     self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                try:
                    net.load_state_dict(state_dict)
                except:
                    print("\t    !! Failed to load model due to a mismatch in the architectures! Initializing anew instead:", net_name)

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def set_requires_grad(self, nets, requires_grad=False):
            """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
            Parameters:
                nets (network list)   -- a list of networks
                requires_grad (bool)  -- whether the networks require gradients or not
            """
            if not isinstance(nets, list):
                nets = [nets]
            for net in nets:
                if net is not None:
                    for param in net.parameters():
                        param.requires_grad = requires_grad

        
    def define_scheduler(self, optimizer, opt):
        lr_policy = opt['lr_policy']
        if lr_policy == 'linear':
            def lambda_rule(epoch):
                lr_l = 1.0 - max(0, epoch + opt.starting_epoch - opt.n_epochs) / float(opt.n_epochs_decay + 1)
                return lr_l
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        elif lr_policy == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt['lr_decay_period'], gamma=opt['lr_decay_ratio'])
        elif lr_policy == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
        elif lr_policy == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
        else:
            return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
        return scheduler

    def compute_weight_for_query_loss(self, y_true, weight_scale=1.0):
        """
        input:
            y_true: [B, T]
        
        output:
            pos_weight: [T]

            https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html#torch.nn.BCEWithLogitsLoss
            if a dataset contains 100 positive and 300 negative examples of a single class, 
            then pos_weight for the class should be equal to 300/100=3.
            The loss would act as if the dataset contains 3x100=300 positive examples.
        """
        with torch.no_grad():
            n_pos = torch.sum(y_true, dim=-1, dtype=torch.int32)
            n_neg = torch.ones_like(n_pos)*y_true.shape[1] - n_pos
            weight = (n_neg / (n_pos + 1e-8)).repeat(y_true.shape[1], 1).transpose(1, 0)
            pos_weight = torch.ones_like(y_true) + y_true * weight * weight_scale
        return pos_weight