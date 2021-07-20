import torch
from torchvision import transforms
import os.path as osp
from pathlib import Path
from torch.utils.data import Dataset
from collections import OrderedDict


class CamVidDataset(Dataset):
    """CamVid dataset for semantic segmentation."""
    mean = [111.376, 63.110, 83.670]
    std = [41.608, 54.237, 68.889]

    mean1 = [0.485, 0.456, 0.406]
    std1 = [0.229, 0.224, 0.225]

    def __init__(self, root, transforms: lambda x: x, pseudo=False, subset='train', iteration=None):
        self.root = root
        self.pseudo = pseudo
        self.subset = subset
        self.images_dir = Path(osp.join(self.root, subset))
        if iteration is not None:
            self.labels_dir = Path(osp.join(self.root, subset + "_pseudoIt" + str(iteration)))
        else:
            self.labels_dir = Path(osp.join(self.root, subset + "_labels"))

        self.images = list(sorted(self.images_dir.glob('*.png')))
        if not pseudo:
            self.labels = list(sorted(self.labels_dir.glob('*.png')))

        self.transforms = transforms

        print("{} images were loaded".format(len(self.images)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        # image = Image.open(self.images[item])
        # label = Image.open(self.labels[item])
        # return self.transforms(image, label)
        ret_dict = {
            'image': self.images[item],
            'subset': self.subset,
            'name': self.images[item].stem
        }
        if not self.pseudo:
            ret_dict['labels'] = self.labels[item]

        return self.transforms(ret_dict)


camvid_colors = OrderedDict([
        (0, (128, 0, 0)),     # 0 Building
        (1, (128, 128, 0)),   # 1 Tree
        (2, (128, 128, 128)), # 2 Sky
        (3, (64, 0, 128)),    # 3 Car
        (4, (192, 128, 128)), # 4 SignSymbol
        (5, (128, 64, 128)),  # 5 Road
        (6, (64, 64, 0)),     # 6 Pedestrian
        (7, (64, 64, 128)),   # 7 Fence
        (8, (192, 192, 128)), # 8 Column_Pole
        (9, (0, 0, 192)),     # 9 Sidewalk
        (10, (0, 128, 192)),  # 10 Bicyclist
        (11, (0, 0, 0))])     # 11 Void
