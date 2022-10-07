import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import PIL

def z_to_str(z):
    if len(z.shape) == 0:
        z = "{:.2f}".format(z)
    else:
        z = ['{:.2f}'.format(zi) for zi in z]
        z = ', '.join(z)
    return z
class Plotter():
    def plot_current_losses(self, save_path, start_epoch, curr_epoch, losses):
        plt.figure(figsize=(10, 4))
        x = np.arange(start_epoch, curr_epoch+1).astype(int)
        lws = []
        for k, l in losses.items():
            if "total" in k.lower():
                lw = 1 * 2
                ls = '-'
            else:
                lw = 0.5 * 2
                ls = '--'
            plt.plot(x, l, linewidth=lw, label=k, linestyle=ls)
            lws.append(lw)
        
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        leg = plt.legend(loc='upper left')
        leg_lines = leg.get_lines()
        for i, lw in enumerate(lws):
            plt.setp(leg_lines[i], linewidth=lw*10)
        leg_texts = leg.get_texts()
        plt.setp(leg_texts, fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.xlabel("Epochs")
        plt.yscale("log")
        plt.title("epoch {}".format(start_epoch+curr_epoch))
        plt.savefig(save_path, dpi=150)
        plt.close()

    def visualize_diff_result(self, save_path, epoch, real_A, real_B, fake_B, title=None):
        n_imgs = real_A.shape[0]
        row_height = 2
        fig = plt.figure(figsize=(row_height*3, row_height*(n_imgs+1)))
        for r in range(n_imgs):
            rA = np.transpose(real_A[r, :, :, :], (1, 2, 0))
            rB = np.transpose(real_B[r, :, :, :], (1, 2, 0))
            fB = np.transpose(fake_B[r, :, :, :], (1, 2, 0))
            diff = rB-fB
            diff = (diff - diff.min()) / (diff.max() - diff.min())
            img = np.clip(np.hstack((rA, rB, fB, diff)), a_min=0, a_max=1)
            ax = fig.add_subplot(n_imgs, 1, r+1)
            ax.imshow(img, cmap="gray")
            
            ax.plot([rA.shape[1], rA.shape[1]], [0, rA.shape[1]], c='w', linewidth=1)
            ax.plot([2*rA.shape[1], 2*rA.shape[1]], [0, rA.shape[1]], c='w', linewidth=1)
            ax.plot([3*rA.shape[1], 3*rA.shape[1]], [0, rA.shape[1]], c='w', linewidth=1)
            ax.set_ylim([0, rA.shape[1]])
            ax.set_xlim([0, 4*rA.shape[1]])
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        title = "Epoch {}".format(epoch)
        plt.suptitle(title)
        plt.savefig(save_path)
        plt.close()


    def visualize_vae_result(self, save_path, title, real_A, real_B, C, z):
        if len(z.shape) == 1:
            z = z[:, None]
        n_imgs = real_A.shape[0]
        row_height = 2
        fig = plt.figure(figsize=(row_height*3, row_height*(n_imgs+1)))
        for r in range(n_imgs):
            rA = np.transpose(real_A[r, :, :, :], (1, 2, 0))
            rB = np.transpose(real_B[r, :, :, :], (1, 2, 0))
            fB = np.transpose(C[r, :, :, :], (1, 2, 0))
            img = np.clip(np.hstack((rA, rB, fB)), a_min=0, a_max=1)
            ax = fig.add_subplot(n_imgs, 1, r+1)
            ax.imshow(img, cmap="gray")
            
            ax.plot([rA.shape[1], rA.shape[1]], [0, rA.shape[1]], c='w', linewidth=1)
            ax.plot([2*rA.shape[1], 2*rA.shape[1]], [0, rA.shape[1]], c='w', linewidth=1)
            ax.set_ylim([0, rA.shape[1]])
            ax.set_xlim([0, 3*rA.shape[1]])
            zi = z[r]
            z_str = "[" 
            for i, z_ in enumerate(zi):
                z_str += "{:.2f}".format(z_)
                if i < len(zi)-1:
                    z_str += ", "
            z_str += "]"
            ax.set_title(z_str, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.suptitle(title, fontsize=16)
        plt.savefig(save_path)
        plt.close()


    def visualize_result(self, save_path, title, real_A, real_B, fake_B):
        n_imgs = real_A.shape[0]
        row_height = 2
        fig = plt.figure(figsize=(row_height*3, row_height*(n_imgs+1)))
        for r in range(n_imgs):
            rA = np.transpose(real_A[r, :, :, :], (1, 2, 0))
            rB = np.transpose(real_B[r, :, :, :], (1, 2, 0))
            fB = np.transpose(fake_B[r, :, :, :], (1, 2, 0))
            img = np.clip(np.hstack((rA, rB, fB)), a_min=0, a_max=1)
            ax = fig.add_subplot(n_imgs, 1, r+1)
            ax.imshow(img, cmap="gray")
            
            ax.plot([rA.shape[1], rA.shape[1]], [0, rA.shape[1]], c='w', linewidth=1)
            ax.plot([2*rA.shape[1], 2*rA.shape[1]], [0, rA.shape[1]], c='w', linewidth=1)
            ax.set_ylim([0, rA.shape[1]])
            ax.set_xlim([0, 3*rA.shape[1]])
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.suptitle(title)
        plt.savefig(save_path)
        plt.close()


    def visualize_overlay_result(self, save_path, epoch, real_A, real_B, fake_B):
        """
        real_A, real_B, fake_B: [B, C, W, H]
        """
        # 1 channel to 3 channels
        mask = np.transpose((np.round(real_A).astype(int) > 0).astype(int), (0, 2, 3, 1))
        mask_invert = np.ones_like(mask) - mask
        real_A = np.clip(np.concatenate((real_A,)*3, axis=1), a_min=0, a_max=1)
        rB = np.clip(np.concatenate((real_B,)*3, axis=1), a_min=0, a_max=1)
        fB = np.clip(np.concatenate((fake_B,)*3, axis=1), a_min=0, a_max=1)

        rA = np.transpose(real_A, (0, 2, 3, 1))
        rB = np.transpose(rB, (0, 2, 3, 1))
        fB = np.transpose(fB, (0, 2, 3, 1))

        # overlay using red
        rB = mask_invert * rB
        fB = mask_invert * fB
        rB[:, :, :, 0] = mask.squeeze()
        fB[:, :, :, 0] = mask.squeeze()

        n_imgs = real_A.shape[0]
        row_height = 2
        fig = plt.figure(figsize=(row_height*3, row_height*(n_imgs+1)))
        for r in range(n_imgs):
            # overlay
            img = np.hstack((rA[r], rB[r], fB[r]))
            ax = fig.add_subplot(n_imgs, 1, r+1)
            ax.imshow(img)
            
            ax.plot([rA.shape[1], rA.shape[1]], [0, rA.shape[1]], c='w', linewidth=1)
            ax.plot([2*rA.shape[1], 2*rA.shape[1]], [0, rA.shape[1]], c='w', linewidth=1)
            ax.set_ylim([0, rA.shape[1]])
            ax.set_xlim([0, 3*rA.shape[1]])
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        title = "Epoch {}".format(epoch)
        plt.suptitle(title)
        plt.savefig(save_path)
        plt.close()