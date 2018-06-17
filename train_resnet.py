import os
import path
import numpy as np

import torch
import argparse

from IPL import Image

from torch import nn
from torch.autograde import Variable 
from torchvision.models as imgnet_models
from utils import *

'''
    The function to get hyper-params from the scripts
'''

def get_args():
    parser = argparse.ArgumentParser("Delf ")
    parser.add_argument('--train_data', type=str)
    parser.add_argument('--test_data', type=str)
    parser.add_argument('--model', type=str, help='model save dir')
    parser.add_argument('--n_classes', type=int, help='num of classes')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--lr_step', type=int, default=200, help='steps of updating of lr')
    parser.add_argument('--gamma', type=float, default=0.1, help='updaterate')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--test_batch_size', type=int, default=64)
    parser.add_argument('--test_iter', type=int, default=1000)
    parser.add_argument('--log', type=str, help='dir of logs')
    parser.add_argument('--save_iter', type=int, default=2000)
    parser.add_argument('--offset', type=int, default=-1)
    parser.add_argument('--shuffle', type=bool, default=True)

    args = parser.parse_args()
    return args
 
def make_model(nclasses):
    resnet50 = image_models.resnet50(pretrained=True)
    for param in resnet50.parameters():
        param.requires_grad = True
    
    #Fix the output nclasses
    resnet50.fc = nn.Linear(2048, nclasses)

    return resnet50


if __name__ == '__main__':
    
    
    
    



