import numpy as np
import os
import json
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from modules.utils.util_io import as_np

def save_image(I, save_path):
    I = (I*255).astype(np.uint8)
    I = Image.fromarray(I)
    I.save(save_path)
    return save_path

def save_reconstructions(save_dir, model, epoch, n_samples=10):
    x = as_np(model.x)
    x_recon = as_np(model.x_recon)
    N = min(n_samples, x.shape[0])
    x = x[0:N]
    x_recon = x_recon[0:N]

    _, n_channel, h, w = x.shape
    vdivider = np.ones((n_channel, h, 1))

    for img_idx in range(N):
        xi = x[img_idx]
        xi_recon = x_recon[img_idx]

        X = xi if img_idx == 0 else np.concatenate((X, vdivider, xi), axis=-1)
        X_recon = xi_recon if img_idx == 0 else np.concatenate((X_recon, vdivider, xi_recon), axis=-1)

    hdivider = np.ones((n_channel, 1, X.shape[-1]))
    img_out = np.transpose(np.concatenate((X, hdivider, X_recon), axis=1), (1, 2, 0)).squeeze()

    save_path = os.path.join(save_dir, "recon_{}.png".format(epoch))
    save_image(img_out, save_path)
    return save_path

def save_losses(save_dir, losses, epoch):
    # save loss values as json
    json_path = os.path.join(save_dir, "losses.json")
    with open(json_path, "w+") as f:
        json.dump(losses, f)

    # save loss values as plot
    plt.figure(figsize=(10, 4))
    matplotlib.rc_file_defaults()
    x_val = np.arange(1, epoch+1).astype(int)
    for loss_name, loss_val in losses.items():
        plt.plot(x_val, loss_val, linewidth=1, label=loss_name)
    leg = plt.legend(loc='upper left')
    leg_lines = leg.get_lines()
    leg_texts = leg.get_texts()
    plt.setp(leg_texts, fontsize=12)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    plt.grid()
    plt.xlabel("Epoch")
    plt.yscale("log")
    plt.title("Train loss at epoch {}".format(epoch))
    save_path = os.path.join(save_dir, "losses.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return json_path, save_path
