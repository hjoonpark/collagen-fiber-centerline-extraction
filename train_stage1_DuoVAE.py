import os, sys, json, shutil

from modules.datasets.collagen_centerline_dataset import CollagenCenterlineDataset
from modules.models.duovae import DuoVAE
from modules.utils.logger import Logger, LogLevel, GPUStat
from modules.utils.util_io import make_directories, load_parameters, set_all_seeds
from modules.utils.util_visualize import save_image, save_reconstructions, save_losses
from modules.utils.util_model import load_model, save_model, get_losses, traverse_y
import torch
import numpy as np

if __name__ == "__main__":
    set_all_seeds(0)

    stage_name = "stage1"

    # make output directories
    dirs = make_directories(root_dir=os.path.join("output", stage_name), sub_dirs=["log", "model", "visualization"])

    # init helper class
    logger = Logger(save_path=os.path.join(dirs["log"], "log.txt"), muted=False)
    gpu_stat = GPUStat()
    logger.print("=============== Training Stage 1 ===============")
    # load user-defined parameters
    params = load_parameters(param_path="parameters/params_stage1.json", copy_to_dir=os.path.join(dirs["log"]))

    # load dataset
    data_dir = params["data_dir"]
    dataset = CollagenCenterlineDataset(data_dir=os.path.join(data_dir, "train"), stage=1, logger=logger)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=params["train"]["batch_size"])

    # init models":
    model = DuoVAE(params=params, is_train=True, logger=logger)
    logger.print(model)
    logger.print("parameters={}".format(json.dumps(params, indent=4)))
    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.print("Trainable parameters={:,}".format(n_trainable_params))

    # log model information
    logger.print("Device={}, GPU Ids={}".format(model.device, model.gpu_ids))
    logger.print("Training on {:,} number of data".format(len(dataset)))

    # make a copy of the model to future reference
    shutil.copyfile("modules/models/duovae.py", os.path.join(dirs["model"], "duovae.py"))

    """
    # To continue training from a saved checkpoint, set load_dir to a directory containing *.pt files   
    # example: load_dir = "output/stage1/model/"
    """
    # load_dir = "output/stage1/model/"
    load_dir = None
    if load_dir is not None:
        load_model(model, load_dir, logger)
    model.train()

    # train
    losses_all = {}
    for epoch in range(1, params["train"]["n_epoch"]+1):
        losses_curr_epoch = {}
        batch_idx = 0
        for batch_idx, data in enumerate(dataloader, 0):
            # ===================================== #
            # main train step
            # ===================================== #
            # set input data
            model.set_input(data)

            # training happens here
            model.optimize_parameters()

            # ===================================== #
            # below are all for plots
            # ===================================== #
            # keep track of loss values
            losses = get_losses(model)
            for loss_name, loss_val in losses.items():
                if loss_name not in losses_curr_epoch:
                    losses_curr_epoch[loss_name] = 0
                losses_curr_epoch[loss_name] += loss_val.detach().cpu().item()

            # save reconstruct results
            if epoch % params["train"]["save_freq"] == 0 and batch_idx == 0:
                save_path = save_reconstructions(save_dir=dirs["log"], model=model, epoch=epoch)
                logger.print("  train recontructions saved: {}".format(save_path))
        
        # check gpu stat
        if epoch == 1:
            logger.print(gpu_stat.get_stat_str())

        # keep track of loss values every epoch
        for loss_name, loss_val in losses_curr_epoch.items():
            if loss_name not in losses_all:
                losses_all[loss_name] = []
            losses_all[loss_name].append(loss_val)

        # log every certain epochs
        do_initial_checks = ((epoch > 0 and epoch <= 50) and (epoch % 10 == 0))
        if do_initial_checks or (epoch % params["train"]["log_freq"] == 0):
            loss_str = "epoch:{}/{} ".format(epoch, params["train"]["n_epoch"])
            for loss_name, loss_vals in losses_all.items():
                loss_str += "{}:{:.4f} ".format(loss_name, loss_vals[-1])
            logger.print(loss_str)
            
        # checkpoint every certain epochs
        if do_initial_checks or (epoch > 0 and epoch % params["train"]["save_freq"] == 0):
            model.eval()
            with torch.no_grad():
                # check gpu stat
                logger.print(gpu_stat.get_stat_str())

                # save loss plot
                json_path, save_path = save_losses(save_dir=dirs["log"], epoch=epoch, losses=losses_all)
                logger.print("  train losses saved: {}, {}".format(json_path, save_path))

                # save model
                save_dir = save_model(save_dir=dirs["model"], model=model)
                logger.print("  model saved: {}".format(dirs["model"]))

                # save y traverse
                traversed_y = traverse_y(model, x=model.x, y=model.y, y_mins=dataset.y_mins, y_maxs=dataset.y_maxs, n_samples=20)
                save_path = save_image(traversed_y.squeeze(), os.path.join(dirs["visualization"], "y_trav_{:05d}.png".format(epoch)))
                logger.print("  y-traverse saved: {}".format(save_path))
            model.train()
    logger.print("=============== DONE ================")