from __future__ import print_function, division
import os
import os.path as osp
import torch
import cv2
import torchvision

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image

class Cell_Dataset(Dataset):
    def __init__(self, data_folder="../../data/cells", transform=None):
        self.data_folder = data_folder
        if not osp.isdir(self.data_folder):
            print("This is not data folder")
            sys.exit()
        list_file = next(os.walk(self.data_folder))[2]
        self.list_file = sorted(list_file)
        self.set_idx = set([int(x[0:3]) for x in list_file])
        self.transform = transform
        print(self.set_idx)

    def __len__(self):
        return len(self.set_idx)

    def __getitem__(self, idx):
        im_name = str(idx).zfill(3) + "cell.png"
        im_path = osp.join(self.data_folder, im_name)
        label_name = str(idx).zfill(3) + "dots.png"
        label_path = osp.join(self.data_folder, label_name)
        im = cv2.imread(im_path, cv2. IMREAD_COLOR)
        label = v2.imread(label_path, cv2. IMREAD_COLOR)
        if self.transform:
            im = self.transform(im)
            label = self.transform(label)
        return im, label

def test_dataloader():
    transform = torchvision.transforms.Compose([
        transforms.Resize(224), 
        transforms.ToTensor(),
    ]) 
    cell_dataset = Cell_Dataset(transform=transform)
    cell_dataloader = torch.utils.data.DataLoader(cell_dataset, batch_size=4, 
                                                    shuffle=True)
    for idx, (images, targets) in enumerate(cell_dataloader):
        print(idx, image.shapes, targets)

if __name__ == "__main__":
    test_dataloader()
