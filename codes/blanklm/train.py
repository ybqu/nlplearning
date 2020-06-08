#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   train.py
@Time    :   2020/06/08 18:51:00
@Author  :   Aiken 
@Version :   1.0
@Contact :   2191002033@cnu.edu.cn
@License :   
@Desc    :   Blank Language Models (BLM)
'''

# here put the import lib


import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torchtext
import argparse
from torchtext.data.utils import get_tokenizer
from models.BLM import BLM


def parse_args():
    """ 
    config setting
    """
    parser = argparse.ArgumentParser(description='train.py')
    parser.add_argument('--gpus', default=[0], type=int, help='device')
    parser.add_argument('--dataset', default='', type=str, choices=[], help='training dataset')
    parser.add_argument('--emsize', default=200, type=int, help='embedding dimension')
    parser.add_argument('--nhidden', default=200, type=int, help='the dimension of the feedforward network model in nn.TranformerEncoder')
    parser.add_argument('--nlayers', default=2, type=int, help='the number of nn.TransformerEncoderLayer in nn.TransformerEncoder')
    parser.add_argument('--nhead', default=2, type=int, help='the number of heads in the multiheadattention models')
    parser.add_argument('--dropout', default=0.2, type=float, help='the dropout value')
    parser.add_argument('--batch_size', default=20, type=int, help='batch_size')
    parser.add_argument('--seed', default=1234, type=int, help='random seed')
    parser.add_argument('--lr', default=3e-5, type=float, help='learning rate')
    parser.add_argument('--num_epochs', default=15, type=int, help='training epochs')
    parser.add_argument('--save_model', action='store_true', help='whether to save the model')
    parser.add_argument('--output_dir', default='./model/', type=str, help='model output path')

    args = parser.parse_args()

    print('args:\n' + args.__repr__())

    return args


# set args as global variables
args = parse_args()
random.seed(args.seed)
np.random.seed(args.seed)

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus


def save_model(path, model):
    """ 
    save model
    """
    if not os.path.exists(path):
        os.mkdir(path)

    # 如果使用多GPU，则使用 model.module.state_dict()
    # model = model.module.state_dict() if len(args.gpus) > 1 else model.state_dict()
    torch.save(model, path)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == "__main__":
    main()