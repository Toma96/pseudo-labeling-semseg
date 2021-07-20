import torch
from torchvision import transforms
import os.path as osp
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class Cityscapes(Dataset):

    mean1 = [0.485, 0.456, 0.406]
    std1 = [0.229, 0.224, 0.225]

    mean = [73.15, 82.90, 72.3]
    std = [47.67, 48.49, 47.73]

    def __init__(self, root, subset='train', has_labels=True, crop_size=(768, 768), ignore_label=255, scale=1):
        self.root = root
        self.images_dir = Path(osp.join(self.root, subset))
        self.labels_dir = Path(osp.join(self.root, subset + '_labels'))

        self.images = list(sorted(self.images_dir.glob('*/*.png')))
        self.labels = list(sorted(self.labels_dir.glob('*/*labelIds.png')))

        self.mean_rgb = tuple(np.uint8(scale * np.array(self.mean)))

        self.resize1 = transforms.Resize(crop_size)
        self.resize2 = transforms.Resize(crop_size, interpolation=Image.NEAREST)
        self.normalize = transforms.Normalize(self.mean, self.std)
        self.tenzoriraj = transforms.ToTensor()

        print('{} images are loaded!'.format(len(self.images)))

    def transforms(self, image, target):
        image = self.resize1(image)
        image = self.tenzoriraj(image)
        image = self.normalize(image)

        target = self.resize2(target)
        target = np.array(target)
        target = torch.tensor(target)  # stvoren na ovaj način kako ga automatski ne bi prebacio 0-1

        return image, target

    def transform(self, image):
        image = self.resize1(image)
        image = self.tenzoriraj(image)
        image = self.normalize(image)
        return image

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image = Image.open(self.images[item])
        label = Image.open(self.labels[item])
        return self.transforms(image, label)


ignore_label = 255
id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                 3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                 7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                 14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                 18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                 28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}


def id2trainId(label, reverse=False):
    label_copy = np.array(label).copy()
    if reverse:
        for v, k in id_to_trainid.items():
            label_copy[label == k] = v
    else:
        for k, v in id_to_trainid.items():
            label_copy[label == k] = v
    return torch.tensor(label_copy)


if __name__ == '__main__':
    dataset = Cityscapes(root="./cityscapes")
    print(dataset[0][0])
    print(dataset[0][1])
    print(id2trainId(dataset[0][1]))
