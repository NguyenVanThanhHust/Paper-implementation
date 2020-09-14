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
from model import EmbeddingNet
import os
import os.path as osp
from torch.utils.tensorboard import SummaryWriter
from data.cell import Cell_Dataset

def train(args):
    adapt = args.adapt
    
    train_transforms = torchvision.transforms.Compose([
        transforms.Resize((255, 255)), 
        transforms.ToTensor(),
    ])
    cell_dataset = Cell_Dataset(transform=train_transforms)

    train_loader = DataLoader(cell_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    net = EmbeddingNet()
    if adapt:
        print("Adapt the adapt module for new dataset")
        for name, param in net.named_parameters():
            if 'adapt' not in name:
                param.require_grad = False
    else:
        print("Train with ILSV2015")

     # Optimizer
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=args.lr_steps)

    criterion = nn.MSELoss()
    net.cuda()
    for epoch in range(args.epochs):
        # Each epoch has a training and validation phase
        net.train()
        log_loss = []

        for i_batch, (input_im, patch_im, input_label) in enumerate(train_loader):
            inputs_im, patch_im, input_label = input_im.cuda(), patch_im.cuda(), input_label.cuda()

            output_heatmap = net(inputs_im, patch_im)
#            print(output_heatmap.shape, input_label.shape)
            loss = criterion(output_heatmap, input_label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            log_loss += [loss.item()]

            if i_batch % 10 == 0:
                log = 'Epoch: %3d, Batch: %5d, ' % (epoch + 1, i_batch)
                log += 'Total Loss: %6.3f, ' % (np.mean(log_loss))
                print(log, datetime.datetime.now())

        scheduler.step()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapt', default=True, type=bool, help='path to train list')
    parser.add_argument('--lr', default=0.001, type=float, help='start learning rate')
    parser.add_argument('--lr_steps', default=[100], type=int, nargs="+", help='epochs to decay learning rate by 10')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='weight decay (default: 1e-5)')
    parser.add_argument('--epochs', default=10, type=int, help='path to train list')
    parser.add_argument('--batch_size', default=2, type=int, help='path to train list')
    parser.add_argument('--num_workers', default=1, type=int, help='path to train list')
    args = parser.parse_args()
    print('Args:')
    for x in vars(args):
        print('\t%s:' % x, getattr(args, x))

    print('Train:')
    train(args=args)
