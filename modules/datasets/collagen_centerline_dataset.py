import os, glob
import numpy as np
import torch
from PIL import Image
from skimage.morphology import skeletonize
import torchvision.transforms as transforms
from skimage import img_as_float, io, img_as_bool
import torchvision.transforms as tf
import random

class CollagenDataset(torch.utils.data.Dataset):
    def __init__(self, rank, data_dir, stage, rand_augment, n_synth_data=0):
        self.stage = stage
        to_tensor = transforms.ToTensor()
        N = -1

        self.filenames = []
        if stage == 1:
            # (Stage 1) collagen centerlines, pre-computed properties
            self.centerlines = []
            self.properties = []
            
            centerline_paths = sorted(glob.glob(os.path.join(data_dir, "labels", "*")))
            for i, cl_path in enumerate(centerline_paths):
                filename = os.path.basename(cl_path).split(".")[0]

                # paths
                prop_path = os.path.join(data_dir, "properties", "{:05d}.txt".format(int(filename)))

                # load
                if ".npy" in cl_path:
                    centerline = to_tensor(np.load(cl_path))
                else:
                    centerline = to_tensor(Image.open(cl_path))

                with open(prop_path, "r") as f:
                    lines = f.readlines()[0]
                    props = torch.from_numpy(np.array([float(v) for v in lines.split(" ")]))[None, :]
                
                self.centerlines.append(centerline)
                self.properties.append(props)
                self.filenames.append(filename)

                if (i-1) % 1000 == 0:
                    print(f"(Stage {stage}) loaded data [{i+1}/{len(centerline_paths)}] [rank {rank}] ")
                if i == N:
                    break

            self.centerlines = torch.cat(self.centerlines, dim=0).float()[:, None, :, :]
            self.properties = torch.cat(self.properties, dim=0).float()
            self.y_mins, self.y_maxs = self.properties.min(dim=0).values, self.properties.max(dim=0).values

            print("self.centerlines: {}, ({:.2f}, {:.2f})".format(self.centerlines.shape, self.centerlines.min(), self.centerlines.max()))
            print("self.properties: {}, ({:.2f}, {:.2f})".format(self.properties.shape, self.properties.min(), self.properties.max()))

        elif stage == 2 or stage == 3:
            # (Stage 2) collagen images, collagen centerlines
            # (Stage 3) collagen centerlines, collagen images
            self.images = []
            self.centerlines = []
            
            centerline_paths = sorted(glob.glob(os.path.join(data_dir, "labels", "*")))
            for i, cl_path in enumerate(centerline_paths):
                filename = os.path.basename(cl_path).split(".")[0]

                # paths
                img_path = os.path.join(data_dir, "images", "{}.png".format(filename))

                # load
                image = to_tensor(Image.open(img_path))
                centerline = to_tensor(Image.open(cl_path))

                self.images.append(image)
                self.centerlines.append(centerline)
                self.filenames.append(filename)

                if (i-1) % 100 == 0:
                    print(f"(Stage {stage}) loaded data [{i+1}/{len(centerline_paths)}] [rank {rank}]")
                if i == N:
                    break

            if stage == 3 and n_synth_data > 0:
                print("Loading synthetic data: {}".format(n_synth_data))
                synth_dir = os.path.join("augmentation_script", "output", "augmentations")

                for aug_type in ["z", "y"]:
                    centerline_paths = sorted(glob.glob(os.path.join(synth_dir, "stage1", f"*{aug_type}*")))

                    centerline_paths = random.sample(centerline_paths, n_synth_data // 2)

                    for i, cl_path in enumerate(centerline_paths):
                        filename = os.path.basename(cl_path)

                        # paths
                        img_path = os.path.join(synth_dir, "stage2", filename)

                        # load
                        image = to_tensor(Image.open(img_path))
                        centerline = to_tensor(Image.open(cl_path))
                        
                        self.images.append(image)
                        self.centerlines.append(centerline)
                        self.filenames.append(filename)

                        if (i-1) % 100 == 0:
                            print(f"(Stage {stage}) loaded {aug_type} synthetic data [{i+1}/{len(centerline_paths)}] [rank {rank}]")

            self.images = torch.cat(self.images, dim=0).float()[:, None, :, :]
            self.centerlines = torch.cat(self.centerlines, dim=0).float()[:, None, :, :]

            # random augmentations
            self.rand_augment = rand_augment
            if rand_augment:
                self.transform = tf.Compose([
                    tf.RandomHorizontalFlip(),
                    tf.RandomRotation(degrees=360, interpolation=tf.InterpolationMode.BILINEAR),
                ])
        else:
            raise NotImplementedError(f"[{self.__class__.__name__ }] Stage number should be either 1, 2, or 3.")

    def __getitem__(self, index):
        data = {"filename": self.filenames[index]}
        if self.stage == 1:
            data["centerline"] = self.centerlines[index]
            data["property"] = self.properties[index]
        elif self.stage == 2 or self.stage == 3:
            image = self.images[index]
            centerline = self.centerlines[index]

            if self.rand_augment:
                image, centerline = self._rand_augment(image, centerline)

            data["image"] = image
            data["centerline"] = centerline
        else:
            raise NotImplementedError(f"[{self.__class__.__name__ }] Stage number should be either 1, 2, or 3.")

        return data

    def __len__(self):
        return len(self.filenames)

    def _rand_augment(self, image, centerline):
        IC = torch.cat((image, centerline), dim=0)
        IC = self.transform(IC)
        I = IC[0].unsqueeze(0)
        C = IC[1].unsqueeze(0)

        # make labels a bit thicker then do skeleton on labels
        C[C>0.1] = 1 # thicker
        C = img_as_float(skeletonize(img_as_bool(C.numpy()))) # skeletonize
        C = torch.FloatTensor(C).to(I.device)

        return I, C