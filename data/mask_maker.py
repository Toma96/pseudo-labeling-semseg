import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import pdb
import numpy as np
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import OrderedDict
import cv2

from .camvid import CamVidDataset, camvid_colors
from .transforms import *


camvid_colors = OrderedDict([
        ((128, 0, 0), 0),     # 0 Building
        ((128, 128, 0), 1),   # 1 Tree
        ((128, 128, 128), 2), # 2 Sky
        ((64, 0, 128), 3),    # 3 Car
        ((192, 128, 128), 4), # 4 SignSymbol
        ((128, 64, 128), 5),  # 5 Road
        ((64, 64, 0), 6),     # 6 Pedestrian
        ((64, 64, 128), 7),   # 7 Fence
        ((192, 192, 128), 8), # 8 Column_Pole
        ((0, 0, 192), 9),     # 9 Sidewalk
        ((0, 128, 192), 10),  # 10 Bicyclist
        ((0, 0, 0), 11)])     # 11 Void


if __name__ == '__main__':
    device = torch.device('cuda')
    batch_size = 1

    trans_train = Compose(
        [Open(),
         Tensor(),
         ]
    )

    dataset_cam_train = CamVidDataset("./camvid", transforms=trans_train, subset='test')
    train_loader_cam = DataLoader(dataset=dataset_cam_train, batch_size=batch_size, shuffle=False)

    im_num = 1
    for ret_dict in train_loader_cam:
        data, name, target = ret_dict['image'].to(device), ret_dict['name'], ret_dict['labels'].to(device)
        subset = ret_dict['subset']

        img_col = target[0]
        h, w = img_col.shape[1], img_col.shape[2]
        mask = torch.zeros(h, w, dtype=torch.uint8)

        print("Image: ", im_num)
        for i in tqdm(range(h)):
            for j in range(w):
                rgb = (img_col[0][i][j].item(), img_col[1][i][j].item(), img_col[2][i][j].item())
                try:
                    camClass = camvid_colors[rgb]
                except KeyError:
                    camClass = 11
                mask[i][j] = camClass

        image = mask.numpy()
        print(image)
        cv2.imwrite("masks/test/{0}.png".format(name[0]), image)
        im_num += 1



