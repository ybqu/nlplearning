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
import time
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import torchtext
import argparse
from torchtext.data.utils import get_tokenizer
from models.BLM import BLM

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
)


def parse_args():
    """
    config setting
    """
    parser = argparse.ArgumentParser(description='train.py')
    parser.add_argument('--gpus', default=[0], type=int, help='device')
    parser.add_argument('--dataset', default='', type=str, choices=[], help='training dataset')
    parser.add_argument('--emsize', default=512, type=int, help='embedding dimension')
    parser.add_argument('--num_hidden', default=2048, type=int, help='the dimension of the feedforward network model in nn.TranformerEncoder')
    parser.add_argument('--num_layers', default=6, type=int, help='the number of nn.TransformerEncoderLayer in nn.TransformerEncoder')
    parser.add_argument('--num_head', default=8, type=int, help='the number of heads in the multiheadattention models')
    parser.add_argument('--dropout', default=0.2, type=float, help='the dropout value')
    parser.add_argument('--batch_size', default=20, type=int, help='batch_size')
    parser.add_argument('--seed', default=1234, type=int, help='random seed')
    parser.add_argument('--lr', default=3e-5, type=float, help='learning rate')
    parser.add_argument('--mr', default=3e-5, type=float, help='mask ratio')
    parser.add_argument('--num_epochs', default=15, type=int, help='training epochs')
    parser.add_argument('--save_model', action='store_true', help='whether to save the model')
    parser.add_argument('--output_dir', default='./model/', type=str, help='model output path')

    args = parser.parse_args()

    print('args:\n' + args.__repr__())

    return args


def train():
    model.train() # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    ntokens = len(TEXT.vocab.stoi)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval = 200
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, len(train_data) // bptt, scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


def evaluate(eval_model, data_source):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    ntokens = len(TEXT.vocab.stoi)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i)
            output = eval_model(data)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)


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
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = BLM(ntokens, args.emsize, args.num_head, args.num_hidden, args.num_layers, args.dropout)
    model.to(device)


if __name__ == "__main__":
    main()