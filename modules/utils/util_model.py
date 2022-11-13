import os
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as dist
from modules.utils.util_io import as_np
from modules.utils.logger import Logger, LogLevel

class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)

def get_available_devices(logger):
    """
    loads whichever available GPUs/CPU
    """
    gpu_ids = []
    device = "cpu"
    if torch.cuda.is_available():
        logger.print("CUDA is available")
        torch.cuda.empty_cache()
        n_gpu = torch.cuda.device_count()
        
        gpu_ids = [i for i in range(n_gpu)]
        device = "cuda:0"
    else:
        """
        support for arm64 MacOS
        https://pytorch.org/blog/introducing-accelerated-pytorch-training-on-mac/
        """
        # this ensures that the current MacOS version is at least 12.3+
        logger.print("CUDA is not available")
        if torch.backends.mps.is_available():
            logger.print("Apple silicon (arm64) is available")
            # this ensures that the current PyTorch installation was built with MPS activated.
            device = "cpu" # discovered using MPS sometimes gives NaN loss - safer not to use it until future updates
            # device = "mps:0"
            gpu_ids.append(0) # (2022) There is no multi-MPS Apple device, yet
            if not torch.backends.mps.is_built():
                logger.print("Current PyTorch installation was not built with MPS activated. Using cpu instead.")
                device = "cpu"

    return device, gpu_ids

def reparameterize(mean, logvar, sample):
    if sample:
        std = torch.exp(0.5*logvar)
        P = dist.Normal(mean, std)
        z = P.rsample()
        return z, P
    else:
        return mean, None

def kl_divergence(Q, P):
    batch_size, z_dim = Q.loc.shape
    return dist.kl_divergence(Q, P).sum()

def save_model(save_dir, model):
    for model_name in model.model_names:
        save_path = os.path.join(save_dir, "{}.pt".format(model_name))

        net = getattr(model, model_name)
        if len(model.gpu_ids) == 1:
            torch.save(net.cpu().state_dict(), save_path)
            net.to(model.device)
        elif len(model.gpu_ids) > 1:
            torch.save(net.module.cpu().state_dict(), save_path)
            net.to(model.gpu_ids[0])
        else:
            torch.save(net.cpu().state_dict(), save_path)
    return save_dir

def load_model(model, load_dir, logger):
    for model_name in model.model_names:
        load_path = os.path.join(load_dir, "{}.pt".format(model_name))
        net = getattr(model, model_name)    
        try:
            logger.print("loading model: {}".format(load_path))
            state_dict = torch.load(load_path, map_location=str(model.device))
            net.load_state_dict(state_dict)
            logger.print("[INFO] model={} loaded succesfully!".format(model_name))
        except:
            logger.print("failed to load - architecture mismatch! Initializing new instead: {}".format(model_name), LogLevel.WARNING.name)

def get_losses(model):
    losses = {}
    for name in model.loss_names:
        losses[name] = getattr(model, "loss_{}".format(name))
    return losses

def traverse_y(model, x, y, y_mins, y_maxs, n_samples):
    x = x.to(model.device)
    y = y.to(model.device)

    unit_range = torch.arange(0, 1+1e-5, 1.0/(n_samples-1))

    (z, _), _ = model.encode(x[[0]], sample=False)

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
            y_new = y_new.to(model.device)

            # encode for w
            w, _ = model.encoder_y(y_new)

            # decode: differs by model
            _, x_recon, _ = model.decode(z, w)
            x_recons = as_np(x_recon) if x_recons is None else np.concatenate((x_recons, vdivider, as_np(x_recon)), axis=-1)
        x_recons_all = x_recons if x_recons_all is None else np.concatenate((x_recons_all, hdivider, x_recons), axis=2)
    x_recons_all = np.transpose(x_recons_all, (0, 2, 3, 1))
    return x_recons_all