import os, sys, json, shutil

from modules.datasets.collagen_centerline_dataset import CollagenCenterlineDataset
from modules.models import DuoVAE, cGAN
from modules.utils.logger import Logger, LogLevel, GPUStat
from modules.utils.util_io import make_directories, load_parameters, set_all_seeds, as_np
from modules.utils.util_visualize import save_image, save_reconstructions, save_losses, images_to_row
from modules.utils.util_model import load_model, save_model, get_losses
import torch
import numpy as np
import argparse

if __name__ == "__main__":
    set_all_seeds(0)

    parser = argparse.ArgumentParser(description='PyTorch implementation of the paper: Variational auto-encoder for collagen fiber centerline generation and extraction in fibrotic cancer tissues.')
    parser.add_argument('stage', type=int, help='Stage number to train: 1, 2, or 3.')
    args = parser.parse_args()
    stage_num = args.stage
    stage_name = "stage{}".format(stage_num)

    # make output directories
    dirs = make_directories(root_dir=os.path.join("output", stage_name), sub_dirs=["log", "model", "validation"])

    # init helper class
    logger = Logger(save_path=os.path.join(dirs["log"], "log.txt"), muted=False)
    gpu_stat = GPUStat()
    logger.print(f"=============== Training Stage {stage_num} ===============")

    # load user-defined parameters
    params = load_parameters(param_path=f"parameters/params_stage{stage_num}.json", copy_to_dir=os.path.join(dirs["log"]))

    # load dataset
    data_dir = params["data_dir"]
    dataset_train = CollagenCenterlineDataset(data_dir=os.path.join(data_dir, "train"), stage=stage_num, logger=logger)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, shuffle=True, batch_size=params["train"]["batch_size"])

    # init model
    if stage_num == 1:
        model_fname = "duovae.py"
        model = DuoVAE(params=params, is_train=True, logger=logger)
    elif stage_num == 2:
        model_fname = "cgan.py"
        model = cGAN(params=params, is_train=True, logger=logger)

        # test data
        dataset_tests = CollagenCenterlineDataset(data_dir=os.path.join(data_dir, "test"), stage=stage_num, logger=logger)
        dataloader_test = torch.utils.data.DataLoader(dataset_tests, shuffle=False, batch_size=1)

    logger.print(model)
    logger.print("parameters={}".format(json.dumps(params, indent=4)))
    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.print("Trainable parameters={:,}".format(n_trainable_params))

    # log information
    logger.print("Device={}, GPU Ids={}".format(model.device, model.gpu_ids))
    logger.print("Training on {:,} number of data".format(len(dataset_train)))

    # make a copy of the model to future reference
    shutil.copyfile(f"modules/models/{model_fname}", os.path.join(dirs["model"], model_fname))

    """
    # To continue training from a saved checkpoint, set load_dir to a directory containing *.pt files   
    # example: load_dir = "output/stage1/model"
    """
    starting_epoch = 1 # For logging purpose. starting_epoch > 1 in case resuming from a checkpoint
    load_dir = None
    # load_dir = "output/stage1/model2"
    if load_dir is not None:
        assert stage_name in load_dir, print("[ERROR] Check if the model being loaded corresponds to the correct stage!")
        load_model(model, load_dir, logger)
    model.train()

    # train
    losses_all = {}
    iters = 0
    for epoch in range(starting_epoch, starting_epoch+params["train"]["n_epoch"]+1):
        losses_curr_epoch = {}
        batch_idx = 0
        for batch_idx, data in enumerate(dataloader_train, 0):
            # ===================================== #
            # main train step
            # ===================================== #
            # set input data
            model.set_input(data)

            # training happens here
            model.optimize_parameters(iters, epoch)

            # ===================================== #
            # below are all for plots
            # ===================================== #
            # keep track of loss values
            losses = get_losses(model)
            for loss_name, loss_val in losses.items():
                if loss_name not in losses_curr_epoch:
                    losses_curr_epoch[loss_name] = 0
                if loss_val is not None:
                    losses_curr_epoch[loss_name] += loss_val.detach().cpu().item()

            # save reconstruct results
            if epoch % params["train"]["save_freq"] == 0 and batch_idx == 0:
                save_path = save_reconstructions(stage=stage_num, save_dir=dirs["log"], model=model, epoch=epoch)
                logger.print("  train recontructions saved: {}".format(save_path))

            iters += 1
            if iters > 1e10:
                iters = 0

        # check gpu stat
        if epoch == 1:
            logger.print(gpu_stat.get_stat_str())

        # keep track of loss values every epoch
        for loss_name, loss_val in losses_curr_epoch.items():
            if loss_name not in losses_all:
                losses_all[loss_name] = []
            losses_all[loss_name].append(loss_val)

        # log losses every some epochs
        do_initial_checks = False #((epoch > 0 and epoch <= 50) and (epoch % 10 == 0))
        if do_initial_checks or (epoch % params["train"]["log_freq"] == 0):
            loss_str = "epoch:{}/{} ".format(epoch, params["train"]["n_epoch"])
            for loss_name, loss_vals in losses_all.items():
                loss_str += "{}:{:.4f} ".format(loss_name, loss_vals[-1])
            logger.print(loss_str)
            
        # checkpoint every some epochs
        if do_initial_checks or (epoch > 0 and epoch % params["train"]["save_freq"] == 0):
            model.eval()
            with torch.no_grad():
                # check gpu stat
                logger.print(gpu_stat.get_stat_str())

                # save loss plot
                json_path, save_path = save_losses(save_dir=dirs["log"], epoch=epoch-starting_epoch+1, losses=losses_all)
                logger.print("  train losses saved: {}, {}".format(json_path, save_path))

                # save model
                save_dir = save_model(save_dir=dirs["model"], model=model)
                logger.print("  model saved: {}".format(dirs["model"]))

                if stage_num == 1:
                    # save y traverse
                    traversed_y = model.traverse_y(x=model.x, y=model.y, y_mins=dataset_train.y_mins, y_maxs=dataset_train.y_maxs, n_samples=20)
                    save_path = save_image(traversed_y.squeeze(), os.path.join(dirs["validation"], "y_trav_{:05d}.png".format(epoch)))
                    logger.print("  y-traverse saved: {}".format(save_path))
                elif stage_num == 2:
                    # cGAN on test images
                    centerlines = []
                    image_recons = []
                    image_trues = []
                    n_samples = 10
                    for i, data in enumerate(dataloader_test, 0):
                        if i >= n_samples:
                            break
                        model.set_input(data)
                        image_recon = model.forward(model.centerline)
                        centerlines.append(as_np(model.centerline))
                        image_recons.append(as_np(image_recon))
                        image_trues.append(as_np(model.image))

                    C = images_to_row(np.array(centerlines).squeeze()[:, None, :, :])
                    I_recon = images_to_row(np.array(image_recons).squeeze()[:, None, :, :])
                    I_true = images_to_row(np.array(image_trues).squeeze()[:, None, :, :])
                    hdivider = np.ones((1, 1, C.shape[-1]))
                    img_out = np.transpose(np.concatenate((C, hdivider, I_recon, hdivider, I_true), axis=1), (1, 2, 0)).squeeze()
                    save_path = os.path.join(dirs["validation"], "val_{}.png".format(epoch))
                    save_image(img_out, save_path)
            model.train()
    logger.print("=============== DONE ================")