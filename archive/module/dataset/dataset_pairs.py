import random
import numpy as np
import os
import torch
import torchvision
from PIL import Image
import glob

def get_image_paths_pairs(A_dir, B_dir=None, val_ratio=0.0, max_dataset_size=-1):
    paths = []
    A_paths = glob.glob(os.path.join(A_dir, "*.png"))
    # random.shuffle(A_paths)

    for A_path in A_paths:
        path_i = {}
        A_name = os.path.basename(A_path)
        path_i["A"] = A_path

        if B_dir is not None:
            B_path = os.path.join(B_dir, A_name)
            if os.path.exists(B_path):
                path_i["B"] = B_path
            else:
                continue
        else:
            continue
        paths.append(path_i)

        if max_dataset_size > 0 and len(paths) >= max_dataset_size:
            break

    n_val = int(len(paths)*val_ratio)
    n_tr = len(paths) - n_val

    paths_tr = paths[0:n_tr]
    paths_val = paths[n_tr:]
    return paths_tr, paths_val

class DatasetPairs(torch.utils.data.Dataset):
    def __init__(self, paths, opt):
        self.paths = paths
        self.opt = opt
        self.img_size = opt['model']['img_size']
        self.resizer = torchvision.transforms.Resize((self.img_size, self.img_size), interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
        self.normlizer = self._gen_normalizer(opt['img_normalize'])

    def __neg1_to_pos1(self, x):
        x = torch.round(x)
        x = (x-x.min()) / (x.max() - x.min())
        x = (x*2) - 1
        return x
    
    def __0_to_1(self, x):
        x = torch.round(x)
        x = (x-x.min()) / (x.max() - x.min())
        return x

    def _gen_normalizer(self, type):
        if type == '-1_to_1':
            return self.__neg1_to_pos1
        elif type == '0_to_1':
            return self.__0_to_1
        else:
            return None

    def _load_img(self, path):
        I = torchvision.transforms.ToTensor()(Image.open(path).convert('L'))

        if I.shape[1] != self.img_size:
            I = self.resizer(I)

        return I
    def __getitem__(self, index):
        output = {}
        for AorB in self.paths[index].keys():
            path = self.paths[index][AorB]
            I = self._load_img(path)
            if self.normlizer is not None:
                I = self.normlizer(I)
            output[AorB] = I
            output["{}_paths".format(AorB)] = path
        
        return output

    def __len__(self):
        return len(self.paths)
