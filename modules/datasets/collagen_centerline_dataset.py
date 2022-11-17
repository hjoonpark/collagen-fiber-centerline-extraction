import os, glob
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms

class CollagenCenterlineDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, stage, logger):
        self.stage = stage
        assert stage == 1 or stage == 2
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
                prop_path = os.path.join(data_dir, "properties", "{}.txt".format(filename))

                # load
                centerline = to_tensor(np.load(cl_path))
                with open(prop_path, "r") as f:
                    lines = f.readlines()[0]
                    props = torch.from_numpy(np.array([float(v) for v in lines.split(" ")]))[None, :]
                
                self.centerlines.append(centerline)
                self.properties.append(props)
                self.filenames.append(filename)

                if (i-1) % 1000 == 0:
                    print("(Stage 1) loaded data [{}/{}]".format(i+1, len(centerline_paths)))
                if i == N:
                    break

            self.centerlines = torch.cat(self.centerlines, dim=0).float()[:, None, :, :]
            self.properties = torch.cat(self.properties, dim=0).float()
            self.y_mins, self.y_maxs = self.properties.min(dim=0).values, self.properties.max(dim=0).values
            
            logger.print("Loaded centerlines={}, {} min/max=({:.2f}, {:.2f}), mean/std=({:.2f}, {:.2f})".format(self.centerlines.shape, self.centerlines.dtype, self.centerlines.min(), self.centerlines.max(), self.centerlines.mean(), self.centerlines.std()))
            logger.print("Loaded properties ={}, {} min/max=({:.2f}, {:.2f}), mean/std=({:.2f}, {:.2f})".format(self.properties.shape, self.properties.dtype, self.properties.min(), self.properties.max(), self.properties.mean(), self.properties.std()))

        elif stage == 2:
            # (Stage 2) collagen images, collagen centerlines
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
                    print("(Stage 2) loaded data [{}/{}]".format(i+1, len(centerline_paths)))
                if i == N:
                    break

            self.images = torch.cat(self.images, dim=0).float()[:, None, :, :]
            self.centerlines = torch.cat(self.centerlines, dim=0).float()[:, None, :, :]
            
            logger.print("Loaded images     ={}, {} min/max=({:.2f}, {:.2f}), mean/std=({:.2f}, {:.2f})".format(self.images.shape, self.images.dtype, self.images.min(), self.images.max(), self.images.mean(), self.images.std()))
            logger.print("Loaded centerlines={}, {} min/max=({:.2f}, {:.2f}), mean/std=({:.2f}, {:.2f})".format(self.centerlines.shape, self.centerlines.dtype, self.centerlines.min(), self.centerlines.max(), self.centerlines.mean(), self.centerlines.std()))

    def __getitem__(self, index):
        data = {"filename": self.filenames[index]}
        if self.stage == 1:
            data["centerline"] = self.centerlines[index]
            data["property"] = self.properties[index]
        elif self.stage == 2:
            data["centerline"] = self.centerlines[index]
            data["image"] = self.images[index]
        return data

    def __len__(self):
        return len(self.filenames)