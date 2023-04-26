import os, sys, json, shutil, glob
sys.path.insert(0, "..") 

from modules.datasets.collagen_centerline_dataset import CollagenDataset
from modules.models import DuoVAE, cGAN, UNet
from modules.utils.logger import Logger, LogLevel, GPUStat
from modules.utils.util_io import make_directories, load_parameters, set_all_seeds, as_np
from modules.utils.util_visualize import save_image, save_reconstructions, save_losses, images_to_row
from modules.utils.util_model import load_model, save_model, get_losses, get_available_devices
from modules.utils.centerline import CenterLine, smooth_mask, iou
from modules.utils.ddp import DDPWraper
from skimage import img_as_float, img_as_ubyte, io
import torch.distributions as dist
import torch.multiprocessing as mp
import torch
import numpy as np
import argparse
import pandas as pd

def export_img(save_path, I):
    I = img_as_ubyte(I.squeeze())
    io.imsave(save_path, I)

def bin_by_frequency(properties, n_bin_intervals=10):
    """
    normalize to N(0, 1)
    """
    y_vals = []
    for prop_idx in range(properties.shape[1]):
        res, bins = pd.qcut(properties[:, prop_idx], q=n_bin_intervals, retbins=True)
        res = list(res.categories.values)

        ys = []
        for i, r in enumerate(res):
            center = (r.left+r.right)*0.5
            ys.append(center)

        ys = np.float32(ys)
        y_vals.append(ys)

    return torch.from_numpy(np.float32(y_vals))

def run(rank, world_size, args):

    out_dir = "output/augmentations"
    os.makedirs(out_dir, exist_ok=1)

    # init helper class
    logger = Logger(save_path=os.path.join(out_dir, "log.txt"), muted=False)
    
    # load user-defined parameters
    params1 = load_parameters(param_path=f"../parameters/params_stage1.json", copy_to_dir=None)
    params2 = load_parameters(param_path=f"../parameters/params_stage2.json", copy_to_dir=None)

    # load models
    device, gpu_ids = get_available_devices(0)
    model1 = DuoVAE(params=params1, is_train=False, device=device)
    model2 = cGAN(params=params2, is_train=False, device=device)

    load_model(model1, "../output/stage1/model4", world_size=1, logger=logger)
    load_model(model2, "../output/stage2/model1", world_size=1, logger=logger)
    model1.eval()
    model2.eval()
    
    dataset_train = CollagenDataset(rank=0, data_dir="../data/base_1188/train", stage=1, rand_augment=False)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=1)
    model1 = model1.to(device)
    model1_module = model1
    model2 = model2.to(device)
    model2_module = model2

    n_y_samples = 5
    y_traverse_points = bin_by_frequency(dataset_train.properties, n_bin_intervals=n_y_samples) # (y_dim, n_traverse_samples)

    single_dir = os.path.join(out_dir, "single")
    stacked_dir = os.path.join(out_dir, "stacked")
    os.makedirs(single_dir, exist_ok=True)
    os.makedirs(stacked_dir, exist_ok=True)

    vdivider = np.ones((1, 1, 256, 1))
    hdivider = np.ones((1, 1, 1, 256*n_y_samples + (n_y_samples-1)))
    hdivider_thick = np.ones((1, 1, 10, 256*n_y_samples + (n_y_samples-1)))
    
    # y-aug
    with torch.no_grad():
        for data_idx, data in enumerate(dataloader_train):
            fname = int(data["filename"][0])

            model1.set_input(data)
            y = model1.y

            # encode z, w
            (z, _), _ = model1.encode(model1.x[[0]], sample=False)

            recons_all = None
            for y_idx in [0, 1, 5, 4, 2, 3]:
                centerline_recons = None
                collagen_imgs = None
                for trav_idx in range(y_traverse_points.shape[1]):
                    # logger.print("[{}/{}] y_idx({}/{}), trav_idx({}/{})".format(fname, len(dataloader_train), y_idx+1, 6, trav_idx+1, y_traverse_points.shape[1]))
                    y_new = torch.clone(y).cpu()

                    y_new[:, y_idx] = y_traverse_points[y_idx, trav_idx]
                    y_new = y_new.to(model1.device)

                    # encode for w
                    w, _ = model1.encoder_y(y_new)

                    # decode
                    _, centerline_recon, _ = model1.decode(z, w)

                    # skeletonize
                    skeletonize = False
                    if skeletonize:
                        centerline_ridge = CenterLine(associate_image=centerline_recon)
                        line_dict = centerline_ridge.ridge_detector(config_fname="modules/utils/ridge_detector_params.json")
                        centerline_ridge = CenterLine(line_dict=line_dict, associate_image=centerline_recon)
                        centerline_recon = centerline_ridge.centerline_image

                        # label -> image
                        centerline_recon = torch.FloatTensor(img_as_float(centerline_recon))[None, None, :, :].to(model2.device)
                        collagen_img = model2.G(centerline_recon)
                        collagen_imgs = as_np(collagen_img) if collagen_imgs is None else np.concatenate((collagen_imgs, vdivider, as_np(collagen_img)), axis=-1)
                    else:
                        # save un-skeletonized
                        save_path = os.path.join(single_dir, "{:05d}_y{:01d}_{:02d}.png".format(fname, y_idx, trav_idx))
                        export_img(save_path, as_np(centerline_recon))
                        logger.print("saved: {}".format(save_path))

                    centerline_recons = as_np(centerline_recon) if centerline_recons is None else np.concatenate((centerline_recons, vdivider, as_np(centerline_recon)), axis=-1)

                if data_idx % 100 == 0:
                    if collagen_imgs is not None:
                        row = np.concatenate((centerline_recons, hdivider, collagen_imgs), axis=2)
                        recons_all = row if recons_all is None else np.concatenate((recons_all, hdivider_thick, row), axis=2)
                    else:
                        row = centerline_recons
                        recons_all = row if recons_all is None else np.concatenate((recons_all, hdivider, row), axis=2)

            if data_idx % 100 == 0:
                # save stacked
                recons_all = np.transpose(recons_all, (0, 2, 3, 1))
                save_path = save_image(recons_all.squeeze(), os.path.join(stacked_dir, "y_trav_{:05d}.png".format(fname)))
                logger.print("saved stacked: {}".format(save_path))
    logger.print("y DONE")
            
    # z-aug
    n_z_samples = 5
    P = None
    with torch.no_grad():
        for data in dataloader_train:
            fname = int(data["filename"][0])

            model1.set_input(data)
            y = model1.y

            # encode z, w
            (z, w), _ = model1.encode(model1.x[[0]], sample=False)

            if P is None:
                P = dist.Normal(torch.zeros_like(z), torch.ones_like(z))

            for sample_idx in range(n_z_samples):
                z = P.sample()

                # decode
                _, centerline_recon, _ = model1.decode(z, w)

                # save un-skeletonized
                save_path = os.path.join(single_dir, "{:05d}_z_{:02d}.png".format(fname, sample_idx))
                export_img(save_path, as_np(centerline_recon))
                logger.print("saved: {}".format(save_path))
    logger.print("z DONE")


if __name__ == "__main__":
    set_all_seeds(0)

    parser = argparse.ArgumentParser(description='PyTorch implementation of the paper: Variational auto-encoder for collagen fiber centerline generation and extraction in fibrotic cancer tissues.')
    args = parser.parse_args()

    # single gpu
    run(0, 1, args) 
