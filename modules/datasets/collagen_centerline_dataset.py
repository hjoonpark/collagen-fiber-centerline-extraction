import os, glob
import numpy as np
import torch


class CollagenCenterlineDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, stage, logger):
        self.stage = stage
        assert stage == 1 or stage == 2
        N = -1

        self.filenames = []
        if stage == 1:
            self.centerlines = []
            self.properties = []
            
            # (Stage 1) DuoVAE takes as inputs: collagen centerlines, pre-computed properties
            img_paths = sorted(glob.glob(os.path.join(data_dir, "images", "*")))
            for i, img_path in enumerate(img_paths):
                filename = os.path.basename(img_path).split(".")[0]

                # paths
                cl_path = os.path.join(data_dir, "labels", "{}.npy".format(filename))
                prop_path = os.path.join(data_dir, "properties", "{}.txt".format(filename))

                # load
                img = np.load(img_path)
                centerline = np.load(cl_path)
                with open(prop_path, "r") as f:
                    lines = f.readlines()[0]
                    props = np.array([float(v) for v in lines.split(" ")])
                
                self.centerlines.append(centerline)
                self.properties.append(props)
                self.filenames.append(filename)

                if (i-1) % 1000 == 0:
                    print("Loaded [{}/{}]".format(i+1, len(img_paths)))
                if i == N:
                    break

            self.centerlines = torch.tensor(np.array(self.centerlines)).float()[:, None, :, :]
            self.properties = torch.tensor(np.array(self.properties)).float()
            self.y_mins, self.y_maxs = self.properties.min(dim=0).values, self.properties.max(dim=0).values
            
            logger.print("Loaded centerlines={}, {} min/max=({:.2f}, {:.2f}), mean/std=({:.2f}, {:.2f})".format(self.centerlines.shape, self.centerlines.dtype, self.centerlines.min(), self.centerlines.max(), self.centerlines.mean(), self.centerlines.std()))
            logger.print("Loaded properties={}, {} min/max=({:.2f}, {:.2f}), mean/std=({:.2f}, {:.2f})".format(self.properties.shape, self.properties.dtype, self.properties.min(), self.properties.max(), self.properties.mean(), self.properties.std()))

        else:
            # (Stage 2)
            self.images = []
            self.centerlines = []
            self.properties = []
            return

    def __getitem__(self, index):
        if self.stage == 1:
            data = {
                "filename": self.filenames[index],
                "centerline": self.centerlines[index],
                "property": self.properties[index]
            }
        return data

    def __len__(self):
        return len(self.filenames)