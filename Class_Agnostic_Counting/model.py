import os
import shutil
import random
import argparse
import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from sklearn.metrics import average_precision_score
from torchvision import transforms, models
from torch.utils.data import DataLoader
import os.path as osp

class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()
        self.base = models.resnet50(pretrained=True)
        self.feature = nn.Sequential(
                    *list(self.base.children())[0:6],
                    # *list(self.base.children())[6][0:2], 
                    # list(self.base.children())[7][2].conv1, 
                    # list(self.base.children())[7][2].bn1, 
                )
        # print(self.feature)

    def forward(self, x):
        x = self.feature(x)
        output = x
        return output

class PatchNet(nn.Module):
    def __init__(self):
        super(PatchNet, self).__init__()
        self.base = BaseNet()
        self.global_max_pool = nn.MaxPool2d(8)

    def forward(self, x):
        x = self.base(x)
        x = self.global_max_pool(x)
        output = x
        return output

class EmbeddingNet(nn.Module):
    def __init__(self, adapt=False):
        super(EmbeddingNet, self).__init__()
        self.base = BaseNet()
        self.patch = PatchNet()

    def forward(self, x1, x2):
        x1 = self.base(x1)
        x2 = self.patch(x2)
        x2 = x2.repeat(1, 1, 32, 32)
        return x1, x2


test_model = EmbeddingNet()
input_1 = torch.rand([4, 3, 255, 255])
input_2 = torch.rand([4, 3, 63, 63])

y1, y2 = test_model(input_1, input_2)

print(y1.shape, y2.shape)
