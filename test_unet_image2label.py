import os, sys
import numpy as np
import torch
from module.utils import load_opt, make_output_folders
from module.dataset.dataset_rand_collagen import RandomCollagen_LessRandom
from module.model._unet_image2label import _unet_image2label
from module.model import as_np
from skimage import io, img_as_ubyte
import warnings

def overlay(A, B):
    """
    overlay A on B
    """
    mask = (A > 0.1).squeeze()
    I = np.concatenate([B, B, B], axis=-1)
    I[mask, 0] = 1
    return I

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    warnings.filterwarnings('ignore')
    warnings.simplefilter('ignore')

    file_number = '50'
    root_dir = os.getcwd()
    output_dir = os.path.join(root_dir, "output/{}_focal_aug_0929".format(file_number))
    out_dirs = make_output_folders(output_dir, ['resTr', 'resTe'])
    
    opt = load_opt('opts/{}.yaml'.format(file_number))

    # initialize models
    model = _unet_image2label(opt['model'], is_train=False)
    load_dir = os.path.join(output_dir, "model/0000030")
    model.load_networks(load_dir)
    model.eval()


    batch_size = 1
    img_dir = opt["img_dir"]
    dataset_tr = RandomCollagen_LessRandom(augment=False, img_path=os.path.join(img_dir, "images_train_aug.npz"), label_path=os.path.join(img_dir, "labels_train_aug.npz"), n_max=-1, device=model.device, logger=None)
    dataset_te = RandomCollagen_LessRandom(augment=False, img_path=os.path.join(img_dir, "images_test_unaug.npz"), label_path=os.path.join(img_dir, "labels_test_unaug.npz"), n_max=-1, device=model.device, logger=None)
    data_loader_tr = torch.utils.data.DataLoader(dataset_tr, shuffle=False, batch_size=batch_size)
    data_loader_te = torch.utils.data.DataLoader(dataset_te, shuffle=False, batch_size=batch_size)

    print('Train {}, Test {}'.format(len(dataset_tr), len(dataset_te)))

    with torch.no_grad():
        """
        forward test dataset first
        """
        for data in data_loader_te:
            # data labelled
            index = data["index"].numpy().squeeze()
            I = data["I"].to(model.device)
            _, L_pred = model(I)
            L_true = data["L"]

            I = img_as_ubyte(as_np(I).squeeze())
            L_pred = img_as_ubyte(as_np(L_pred).squeeze())
            L_true = img_as_ubyte(as_np(L_true).squeeze())

            save_dir = out_dirs["resTe"]
            io.imsave(os.path.join(save_dir, "{}_I.png".format(index)), I)
            io.imsave(os.path.join(save_dir, "{}_Lpred.png".format(index)), L_pred)
            io.imsave(os.path.join(save_dir, "{}_Ltrue.png".format(index)), L_true)
            print("test dataset [{}/{}]".format(index, len(dataset_te)))
        """
        forward train dataset first
        """
        for data in data_loader_tr:
            # data labelled
            index = data["index"].numpy().squeeze()
            I = data["I"].to(model.device)
            _, L_pred = model(I)
            L_true = data["L"]

            I = img_as_ubyte(as_np(I).squeeze())
            L_pred = img_as_ubyte(as_np(L_pred).squeeze())
            L_true = img_as_ubyte(as_np(L_true).squeeze())

            save_dir = out_dirs["resTr"]
            io.imsave(os.path.join(save_dir, "{}_I.png".format(index)), I)
            io.imsave(os.path.join(save_dir, "{}_Lpred.png".format(index)), L_pred)
            io.imsave(os.path.join(save_dir, "{}_Ltrue.png".format(index)), L_true)
            print("train dataset [{}/{}]".format(index, len(dataset_tr)))
    print("@@@@@@ DONE @@@@@@@@@@")
