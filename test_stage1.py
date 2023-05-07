
import os, sys, json, shutil, glob

from modules.datasets.collagen_centerline_dataset import CollagenDataset
from modules.models import DuoVAE, cGAN, UNet
from modules.utils.logger import Logger, LogLevel, GPUStat
from modules.utils.util_io import make_directories, load_parameters, set_all_seeds, as_np
from modules.utils.util_visualize import save_image, save_reconstructions, save_losses, images_to_row
from modules.utils.util_model import load_model, save_model, get_losses, get_available_devices
from modules.utils.ddp import DDPWraper
import torch.multiprocessing as mp
import torch
import numpy as np
import argparse

def run(args):
    model_dir = args.model_dir

    # make output directories
    out_dir = os.path.join(model_dir, "..", "trainset")
    os.makedirs(out_dir, exist_ok=True)

    dirs = make_directories(root_dir=out_dir, sub_dirs=["stage1"])

    # init helper class
    logger = Logger(save_path=os.path.join(out_dir, "log.txt"), muted=False)
    logger.print(f"input arguments: {args}")

    device = "cuda:0"

    # load user-defined parameters
    params = load_parameters(param_path=f"parameters/params_stage1.json", copy_to_dir=None)
    batch_size = 1
    data_dir = "data/base_1188"

    # test data
    dataset_tests = CollagenDataset(rank=0, data_dir=os.path.join(data_dir, "train"), stage=1, rand_augment=False)
    dataloader_test = torch.utils.data.DataLoader(dataset_tests, shuffle=False, batch_size=1)

    # init model
    model = DuoVAE(params=params, is_train=False, device=device)

    # load saved model
    if len(glob.glob(os.path.join(model_dir, "*.pt"))) > 0:
        load_model(model, model_dir, 1, logger)
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        # UNet on test images
        for i, data in enumerate(dataloader_test, 0):
            x = data["centerline"].to(device)
            (z, w), _ = model.encode(x, sample=False)
            _, centerline_recon, _ = model.decode(z, w)
            centerline_recon = as_np(centerline_recon).squeeze()

            filename = "{:03d}".format(int(data["filename"][0]))

            save_path = os.path.join(dirs["stage1"], f"{filename}.png")
            save_image(centerline_recon, save_path)
            logger.print("[{}/{}] saved: {}".format(i+1, len(dataloader_test), save_path))
    
    logger.print("===== DONE =====")


if __name__ == "__main__":
    set_all_seeds(0)

    parser = argparse.ArgumentParser(description='PyTorch implementation of the paper: Variational auto-encoder for collagen fiber centerline generation and extraction in fibrotic cancer tissues.')
    parser.add_argument('model_dir', type=str, help='Model directory where .pt file are located.', default=None)
    args = parser.parse_args()

    run(args) 
