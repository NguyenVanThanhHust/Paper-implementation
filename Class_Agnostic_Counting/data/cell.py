from __future__ import print_function, division
import os
import sys
import os.path as osp
import torch
import cv2
import torchvision
import random as rd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
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
        self.patch_transform = torchvision.transforms.Compose([
            transforms.ToTensor(),
        ])
        # print(self.set_idx)

    def __len__(self):
        return len(self.set_idx) - 1

    def __getitem__(self, idx):
        im_name = str(idx + 1).zfill(3) + "cell.png"
        im_path = osp.join(self.data_folder, im_name)
        label_name = str(idx + 1).zfill(3) + "dots.png"
        label_path = osp.join(self.data_folder, label_name)
        im = Image.open(im_path).convert('RGB')
        label = Image.open(label_path).convert('L')
        im_size = label.size 

        nonzero_x, nonzero_y = np.nonzero(label)
        rd_index = rd.randint(0, len(nonzero_x) - 1)
        patch_x, patch_y = nonzero_x[rd_index], nonzero_y[rd_index]
        while not is_in_center_area(patch_x, patch_y, im_size):
            rd_index = rd.randint(0, len(nonzero_x) - 1)
            patch_x, patch_y = nonzero_x[rd_index], nonzero_y[rd_index]

        bbox = [patch_x  -32, patch_y - 32, patch_x + 31, patch_y + 31]
        patch_img = im.crop(bbox)
        
        label = label.resize((63, 63))
        if self.transform:
            im = self.transform(im)
            patch_img = self.patch_transform(patch_img)
        to_tensor = transforms.ToTensor()
        label = to_tensor(label)
        return im, patch_img, label

def is_in_center_area(patch_x, patch_y, im_size):
    width, height = im_size
    if patch_x > width*0.8 or patch_x < width*0.2:
        return False
    if patch_y > height*0.8 or patch_y < height*0.2:
        return False
    return True

def test_dataloader():
    transform = torchvision.transforms.Compose([
        transforms.Resize((255, 255)), 
        transforms.ToTensor(),
    ])
     
    cell_dataset = Cell_Dataset(transform=transform)
    cell_dataloader = torch.utils.data.DataLoader(cell_dataset, batch_size=4, 
                                                    shuffle=True)
    for idx, (patch_img, images, targets) in enumerate(cell_dataloader):
        print(idx, patch_img.shape, images.shape, targets.shape)
        if idx > 0:
            break

#if __name__ == "__main__":
#    test_dataloader()
