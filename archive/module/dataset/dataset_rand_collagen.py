import numpy as np
import os
import torch
import torchvision.transforms as tf
import glob
from skimage.morphology import skeletonize
from skimage import img_as_float, io, img_as_bool
import random
import skimage

class RandomCollagen(torch.utils.data.Dataset):
    def __init__(self, opt, logger):
        self.transform = tf.Compose([
            tf.RandomHorizontalFlip(),
            tf.RandomRotation(degrees=360, interpolation=tf.InterpolationMode.BILINEAR),
            tf.RandomCrop([256, 256])
        ])

        self.opt = opt
        n_max = opt["max_dataset_size"]
        img_paths = sorted(glob.glob(os.path.join(opt["img_dir"], "images/*.png")))[:n_max]
        label_dir = os.path.join(opt["img_dir"], "labels")

        self.Ls = []
        self.Is = []
        for I_path in img_paths:
            fname = os.path.basename(I_path)
            I = torch.FloatTensor(img_as_float(io.imread(I_path)))[None, :, :]

            L_path = os.path.join(label_dir, fname)
            L = torch.FloatTensor(img_as_float(io.imread(L_path)))[None, :, :]

            self.Is.append(I)
            self.Ls.append(L)

        # sanity checks
        logger.write("{} images, {} labels loaded".format(len(self.Is), len(self.Ls)))
        N = 5
        logger.write(">> sanity check on random {} images".format(N))
        for _ in range(N):
            I = random.choice(self.Is)
            print("  size={}, min/max=({:.2f}, {:.2f}), dtype={}".format(I.shape, I.min(), I.max(), I.dtype))

        logger.write(">> sanity check on random {} labels".format(N))
        for _ in range(N):
            L = random.choice(self.Ls)
            print("  size={}, min/max=({:.2f}, {:.2f}), dtype={}".format(L.shape, L.min(), L.max(), L.dtype))

    def __len__(self):
        return len(self.Is)

    def __getitem__(self, i):
        I = self.Is[i] # (1, W, H)
        L = self.Ls[i] # (1, W, H)
        IL = torch.cat((I, L), dim=0)
        IL = self.transform(IL)
        I = IL[0].unsqueeze(0)
        L = IL[1].unsqueeze(0)
        return {"I": I, "L": L}
         
class RandomCollagen_LessRandom(torch.utils.data.Dataset):
    def __init__(self, normalize_type, augment, extra_aug_img, img_path, label_path, n_max, device, logger):
        """
        input images are 256x256
        """
        self.augment = augment
        self.extra_aug_img = extra_aug_img
        if augment:
            self.transform = tf.Compose([
                tf.RandomHorizontalFlip(),
                tf.RandomRotation(degrees=360, interpolation=tf.InterpolationMode.BILINEAR),
            ])

        self.device = device
        
        self.Is = torch.FloatTensor(np.load(img_path)["a"])[:, None, :, :]
        self.Ls = torch.FloatTensor(np.load(label_path)["a"])[:, None, :, :]


        if n_max > 0:
            self.Is = self.Is[0:n_max, :, :, :]
            self.Ls = self.Ls[0:n_max, :, :, :]

        self.indices = np.arange(0, len(self.Is))

        # sanity checks
        if logger is not None:
            logger.write("{} images, {} labels loaded".format(self.Is.shape, self.Ls.shape))
        else:
            print("{} images, {} labels loaded".format(self.Is.shape, self.Ls.shape))
        N = 5
        print(">> sanity check on random {} images/labels".format(N))
        rand_indices = np.random.choice(np.arange(len(self.Is)), N)
        for i in rand_indices:
            I = self.Is[i]
            L = self.Ls[i]
            print("  [{}] Image: size={}\tmin/max=({:.2f}, {:.2f}), dtype={}".format(i, I.shape, I.min(), I.max(), I.dtype))
            print("       Label: size={}\tmin/max=({:.2f}, {:.2f}), dtype={}".format(L.shape, L.min(), L.max(), L.dtype))

        self.normalize_type = normalize_type
        if logger is not None:
            logger.write("Image normalization type: {}".format(normalize_type))
        else:
            print("Image normalization type: {}".format(normalize_type))
    def __len__(self):
        return len(self.Is)

    def __getitem__(self, i):
        index = self.indices[i]

        I = self.Is[i, :, :, :] # (1, W, H)
        L = self.Ls[i, :, :, :] # (1, W, H)

        if self.augment:
            IL = torch.cat((I, L), dim=0)
            IL = self.transform(IL)
            I = IL[0].unsqueeze(0)
            L = IL[1].unsqueeze(0)

            # make labels a bit thicker then do skeleton on labels
            L[L>0.1] = 1 # thicker
            L = img_as_float(skeletonize(img_as_bool(L.numpy()))) # skeletonize
            L = torch.FloatTensor(L).to(I.device)

            # do gamma correction & add random noise by 90% chance
            if self.extra_aug_img:
                do_additional = random.uniform(0, 1) > 0.1
                if do_additional:
                    # gamma correction
                    gamma = random.uniform(0.3, 1.5)
                    I = tf.functional.adjust_gamma(I, gamma=gamma)

                    # random noise: gaussian, poisson
                    noise_mode = random.choice([0, 1])
                    if noise_mode == 0:
                        # gaussian
                        var = random.uniform(0.001, 0.02+1e-5)
                        I = torch.FloatTensor(skimage.util.random_noise(I, mode='gaussian', var=var)).to(I.device)
                    elif noise_mode == 1:
                        # poisson
                        I = torch.FloatTensor(skimage.util.random_noise(I, mode='poisson')).to(I.device)
                    else:
                        assert False
                
        if self.normalize_type == 'tanh':
            # make Is [-1, 1] to use nn.Tanh()
            I = (2*I-1)
        return {"index": index, "I": I, "L": L}