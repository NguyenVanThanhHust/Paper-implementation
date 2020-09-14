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
                )
        # print(self.feature)

    def forward(self, x):
#        print(x.shape)
        x = self.feature(x)
        output = x
        return output

class PatchNet(nn.Module):
    def __init__(self):
        super(PatchNet, self).__init__()
        self.base = BaseNet()
        self.global_max_pool = nn.MaxPool2d(8)

    def forward(self, x):
#        print(x.shape)
        x = self.base(x)
        x = self.global_max_pool(x)
        output = x
        return output

class AdaptModule(nn.Module):
    def __init__(self, ):
        super(AdaptModule, self).__init__()
        self.conv = nn.Conv2d(1024, 256, (3, 3), padding=(1, 1))
        self.batch_norm_1 = nn.BatchNorm2d(num_features=256)
        self.relu1 = nn.ReLU()
        
        self.convt = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.batch_norm_2 = nn.BatchNorm2d(num_features=256)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu1(self.batch_norm_1(x))
        x = self.convt(x)
        x = self.relu2(self.batch_norm_2(x))
        return x

class EmbeddingNet(nn.Module):
    def __init__(self, adapt=False):
        super(EmbeddingNet, self).__init__()
        self.base = BaseNet()
        self.patch = PatchNet()
        self.adapt_module = AdaptModule()
        self.predict = nn.Conv2d(256, 1, (3, 3), padding=(1, 1))

    def forward(self, x1, x2):
        x1 = self.base(x1)
        x2 = self.patch(x2)
#        print(x1.shape)
#        print(x2.shape)
        x2 = x2.repeat(1, 1, 32, 32)
        x = torch.cat([x1, x2], dim=1)
        x = self.adapt_module(x)
        y = self.predict(x)
        return y


#test_model = EmbeddingNet()
#input_1 = torch.rand([4, 3, 255, 255])
#input_2 = torch.rand([4, 3, 63, 63])
#
#y= test_model(input_1, input_2)
#
#print(y.shape) # torch.Size([4, 1, 63, 63])

