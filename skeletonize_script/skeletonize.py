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
import warnings

def export_img(save_path, I):
    I = img_as_ubyte(I.squeeze())
    io.imsave(save_path, I)

def run(args):
    warnings.filterwarnings('ignore')
    warnings.simplefilter('ignore')
    
    in_dir = args.input_dir
    out_dir = os.path.join(in_dir, "..", "skeletonized")
    os.makedirs(out_dir, exist_ok=True)
    thread_idx = args.thread_idx

    # init helper class
    logger = Logger(save_path=os.path.join(out_dir, "..", f"log_skeletonize_{thread_idx}.txt"), muted=False)
    logger.print(f"input arguments: {args}")


    img_paths = sorted(glob.glob(os.path.join(in_dir, "*.png")))

    start_idx = args.start_idx # inclusive
    end_idx = args.end_idx # non inclusive

    if end_idx == -1:
        end_idx = len(img_paths)

    for i in range(start_idx, end_idx):
        img_path = img_paths[i]

        fname = os.path.basename(img_path)
        I = img_as_float(io.imread(img_path))

        centerline_ridge = CenterLine(associate_image=I)
        line_dict = centerline_ridge.ridge_detector(config_fname="../modules/utils/ridge_detector_params.json")
        centerline_ridge = CenterLine(line_dict=line_dict, associate_image=I)
        centerline_recon = centerline_ridge.centerline_image

        save_path = os.path.join(out_dir, fname)
        export_img(save_path, centerline_recon)

        logger.print("[{}/{} from {}] saved: {}".format(i+1, end_idx-start_idx, len(img_paths), save_path))

    logger.print("==== DONE ====")

if __name__ == "__main__":
    set_all_seeds(0)

    parser = argparse.ArgumentParser(description='PyTorch implementation of the paper: Variational auto-encoder for collagen fiber centerline generation and extraction in fibrotic cancer tissues.')
    parser.add_argument('input_dir', type=str, help='Input directory', default=None)
    parser.add_argument('--thread-idx', type=int, help='thread index', default=0)
    parser.add_argument('--start-idx', type=int, help='start index', default=0)
    parser.add_argument('--end-idx', type=int, help='end index', default=-1)
    args = parser.parse_args()

    # single gpu
    run(args) 
