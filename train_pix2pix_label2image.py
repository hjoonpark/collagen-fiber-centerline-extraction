import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import numpy as np
import torch
import torch.nn.functional as F
import math
from module.utils import Logger, Plotter, load_opt, save_opt, make_output_folders
from module.dataset.dataset_rand_collagen import RandomCollagen_LessRandom
from module.model._pix2pix_label2image import _pix2pix
from module.model import as_np
import matplotlib.pyplot as plt
import shutil
from skimage import img_as_ubyte

def overlay(A, B):
    mask = (A > 0).squeeze()
    I = np.concatenate([B, B, B], axis=-1)
    I[mask, 0] = 1
    return img_as_ubyte(I)

def visualize_res(save_path, suptitle, res):
    rAs = res['real_L']
    rBs = res['real_I']
    fBs = res['fake_I']
    index = res["index"]

    B = rAs.shape[0]
    
    n_rows = min(10, B)
    n_cols = 5
    
    s = 2
    fig = plt.figure(figsize=(s*n_cols, s*n_rows))
    for r in range(n_rows):
        i = r*n_cols
        ax = fig.add_subplot(n_rows, n_cols, i+1)
        rA = rAs[r].squeeze()
        ax.imshow(rA, cmap='gray')
        if r == 0:
            ax.set_title("[{}] Input L {}\n({:.2f}, {:.2f})".format(index[r], rA.shape, rA.min(), rA.max()))
        else:
            ax.set_title("[{}] L ({:.2f}, {:.2f})".format(index[r], rA.min(), rA.max()))
        ax = fig.add_subplot(n_rows, n_cols, i+2)
        fB = fBs[r].squeeze()
        ax.imshow(fB, cmap='gray')
        ax.set_title("Fake I ({:.2f}, {:.2f})".format(fB.min(), fB.max()))

        ax = fig.add_subplot(n_rows, n_cols, i+3)
        rB = rBs[r].squeeze()
        ax.imshow(rB, cmap='gray')
        ax.set_title("Real I: ({:.2f}, {:.2f})".format(rB.min(), rB.max()))

        ax = fig.add_subplot(n_rows, n_cols, i+4)
        overlay_fB = overlay(fB, rA[:, :, None])
        ax.imshow(overlay_fB)
        ax.set_title("Overlay (Fake I)")

        ax = fig.add_subplot(n_rows, n_cols, i+5)
        overlay_rB = overlay(rB, rA[:, :, None])
        ax.imshow(overlay_rB)
        ax.set_title("Overlay (Real I)")

    plt.suptitle(suptitle)
    plt.subplots_adjust(top=0.90)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    file_number = '27'
    root_dir = os.getcwd()
    output_dir = os.path.join(root_dir, "output/{}".format(file_number))
    os.makedirs(output_dir, exist_ok=True)

    out_dirs = make_output_folders(output_dir, ['log', 'model'])

    log_path = os.path.join(out_dirs['log'], 'loss_log.txt')
    plot_path = os.path.join(out_dirs['log'], "loss_plot.jpg")
    
    opt = load_opt('opts/{}.yaml'.format(file_number))
    save_opt(os.path.join(out_dirs['log'], 'opt.yaml'), opt)
    shutil.copy(os.path.join(root_dir, 'module/model/_pix2pix_label2image.py'), out_dirs['model'])

    logger = Logger(log_path)
    plotter = Plotter()

    # initialize models
    model = _pix2pix(opt['model'], is_train=True)

    if opt['starting_epoch'] > 0:
        fname = str(opt['starting_epoch'])
        while len(fname) < 5:
            fname = '0' + fname
        load_dir = os.path.join(output_dir, "{}".format(fname))
    else:
        load_dir = None
    if load_dir is not None:
        model.load_networks(load_dir, logger)
    logger.write(model)

    batch_size = opt['batch_size']
    unaug_dir = opt["img_dir"]
    normalize_type = opt['normalize_type']
    dataset_tr = RandomCollagen_LessRandom(normalize_type=normalize_type, augment=True, extra_aug_img=False, img_path=os.path.join(unaug_dir, "images_train_unaug.npz"), label_path=os.path.join(unaug_dir, "labels_train_unaug.npz"), n_max=-1, device=model.device, logger=logger)
    dataset_te = RandomCollagen_LessRandom(normalize_type=normalize_type, augment=False, extra_aug_img=False, img_path=os.path.join(unaug_dir, "images_test_unaug.npz"), label_path=os.path.join(unaug_dir, "labels_test_unaug.npz"), n_max=-1, device=model.device, logger=logger)
    data_loader_tr = torch.utils.data.DataLoader(dataset_tr, shuffle=True, batch_size=batch_size)
    data_loader_te = torch.utils.data.DataLoader(dataset_te, shuffle=False, batch_size=batch_size)
    logger.write("Dataset: train({}) test({}) | batch_size={}".format(len(dataset_tr), len(dataset_te), batch_size))
 
    n_data = len(dataset_tr)
    losses_per_epoch = {}
    epoch0 = opt['starting_epoch']
    iters = 0
    for epoch in range(epoch0, epoch0 + opt['n_epochs'] + 1):
        model.train()
        loss_curr = {}

        save_plots = (epoch-epoch0) == 0 or epoch % opt['val_freq'] == 0
        batch_idx = 0
        for data in data_loader_tr:
            # data labelled
            model.set_input(data)
            model.optimize_parameters(iters)
            losses = model.get_current_losses()
            n_minibatch = data["index"].shape[0]

            losses = model.get_current_losses()
            for k, v in losses.items():
                if k not in loss_curr:
                    loss_curr[k] = v * n_minibatch / n_data
                else:
                    loss_curr[k] += v * n_minibatch / n_data

            if batch_idx == 0 and save_plots:
                # keep images
                N = min(10, batch_size)
                rL = as_np(model.real_L)[0:N]
                rI = as_np(model.real_I)[0:N]
                fI = as_np(model.fake_I)[0:N]
                index = model.index[0:N]
                imgs_for_plot = {"index": index, "real_L": rL, "real_I": rI, "fake_I": fI}
                
            batch_idx += 1
            if batch_idx > opt['iter_per_epoch']: break

            iters += 1
            if iters > 1e5:
                iters = 0
        # batch iteration ends
        for k, v in loss_curr.items():
            if k not in losses_per_epoch:
                losses_per_epoch[k] = []
            losses_per_epoch[k].append(v)
        if epoch % opt['print_freq'] == 0:
            logger.print_current_losses(epoch, opt['starting_epoch'] + opt['n_epochs'], loss_curr)
        
        # break if explodes
        for k, v in losses.items():
            if math.isnan(v):
                logger.write("nan detected: {}:{}".format(k, v))
                assert (not math.isnan(v)), "NaN detected!"

        # validation
        if (epoch-epoch0) == 0 or epoch % opt['val_freq'] == 0:
            save_path = os.path.join(out_dirs['log'], 'tr_{}.jpg'.format(epoch))
            visualize_res(save_path, 'Train epoch {}'.format(epoch), imgs_for_plot)

            model.eval()
            with torch.no_grad():       
                for data in data_loader_te:
                    fI = model.forward(data['L'].to(model.device))

                    N = min(10, min(len(dataset_te), batch_size))
                    rL = as_np(data['L'])[0:N]
                    rI = as_np(data['I'])[0:N]
                    fI = as_np(fI)[0:N]
                    index = data["index"][0:N]
                    imgs_for_plot = {"index": index, "real_L": rL, "real_I": rI, "fake_I": fI}

                    save_path = os.path.join(out_dirs['log'], 'val_{}.jpg'.format(epoch))
                    visualize_res(save_path, 'Validate epoch {}'.format(epoch), imgs_for_plot)
            model.train()

        if (epoch - epoch0 == 30) or (epoch > 0 and epoch % opt['chkpt_freq'] == 0):
            
            plotter.plot_current_losses(plot_path, epoch0, epoch, losses_per_epoch)
            logger.write("- {}".format(plot_path))

            # save model
            model_dir_epoch = os.path.join(out_dirs['model'], "{:05d}".format(epoch))
            os.makedirs(model_dir_epoch, exist_ok=True)
            model.save_networks(model_dir_epoch)
            logger.write("- Model saved: {}".format(model_dir_epoch))

            # only keep recent 10 models
            folders_names = sorted(next(os.walk(out_dirs['model']))[1])
            if len(folders_names) > 5:
                f_to_delete = os.path.join(out_dirs['model'], folders_names[0])
                shutil.rmtree(f_to_delete)
                logger.write("- Folder removed: {}".format(folders_names[0]))

    logger.write("@@@@@@ DONE @@@@@@@@@@")