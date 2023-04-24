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
from PIL import Image
import torch.distributions as dist
import torch.multiprocessing as mp
import torchvision
import torch
import numpy as np
import argparse
import pandas as pd
import warnings

def export_img(save_path, I):
    I = img_as_ubyte(I.squeeze())
    io.imsave(save_path, I)

def run(threa_idx, args):
    warnings.filterwarnings('ignore')
    warnings.simplefilter('ignore')

    out_dir = "output/augmentations"
    stage1_dir = os.path.join(out_dir, 'stage1')
    stage2_dir = os.path.join(out_dir, 'stage2')
    os.makedirs(stage1_dir, exist_ok=1)
    os.makedirs(stage2_dir, exist_ok=1)

    # init helper class
    logger = Logger(save_path=os.path.join(out_dir, "log_stage2_thread{}.txt".format(threa_idx)), muted=False)
    
    # load user-defined parameters
    params2 = load_parameters(param_path=f"../parameters/params_stage2.json", copy_to_dir=None)

    # load models
    device, gpu_ids = get_available_devices(0)
    model2 = cGAN(params=params2, is_train=False, device=device)

    load_model(model2, "../output/stage2/model1", world_size=1, logger=logger)
    model2.eval()
    
    model2 = model2.to(device)
    model2_module = model2

    single_dir = os.path.join(out_dir, "single")

    x_paths_all = glob.glob(os.path.join(single_dir, "*.png"))
    x_paths = []
    for x_path in x_paths_all:
        data_idx = int(os.path.basename(x_path).split("_")[0])
        if args.start_idx <= data_idx and data_idx < args.end_idx:
            x_paths.append(x_path)
    x_paths = sorted(x_paths)
    logger.print("processing {} data. Index [{}, {})".format(len(x_paths), args.start_idx, args.end_idx))

    to_tensor = torchvision.transforms.Compose([
        torchvision.transforms.PILToTensor()
    ])
    for x_path in x_paths:
        basename = os.path.basename(x_path)

        # decode
        centerline_recon = to_tensor(Image.open(x_path))

        centerline_ridge = CenterLine(associate_image=centerline_recon)
        line_dict = centerline_ridge.ridge_detector(config_fname="../modules/utils/ridge_detector_params.json")
        centerline_ridge = CenterLine(line_dict=line_dict, associate_image=centerline_recon)
        centerline_recon = img_as_float(centerline_ridge.centerline_image)

        # label -> image
        centerline_recon = torch.FloatTensor(centerline_recon)[None, None, :, :].to(model2.device)
        collagen_img = model2.G(centerline_recon)
        collagen_img = (collagen_img+1)/2

        # save stage1
        save_path = os.path.join(stage1_dir, basename)
        export_img(save_path, as_np(centerline_recon))

        # save stage2
        save_path = os.path.join(stage2_dir, basename)
        export_img(save_path, as_np(collagen_img))

        logger.print("saved: {}".format(save_path))

    logger.print("stage2 DONE")


if __name__ == "__main__":
    set_all_seeds(0)

    parser = argparse.ArgumentParser(description='PyTorch implementation of the paper: Variational auto-encoder for collagen fiber centerline generation and extraction in fibrotic cancer tissues.')
    parser.add_argument("--worker-idx", type=int, help="worker index", default=0)
    parser.add_argument("--start-idx", type=int, help="starting index", default=0)
    parser.add_argument("--end-idx", type=int, help="end index", default=0)
    args = parser.parse_args()

    run(args.worker_idx, args)
