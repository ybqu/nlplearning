#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2020/07/13 22:40:21
@Author  :   Aiken 
@Version :   1.0
@Contact :   2191002033@cnu.edu.cn
@License :   
@Desc    :   None
'''

# here put the import lib
import torch
import math
import random


def dataprocess(corpus_file, save_path, mr=0.2):
    """ 数据处理 """
    sents = []
    with open(corpus_file, 'r') as r:
        lines = r.readlines()
        for line in lines:
            sent_list = line.strip().split(' ')
            try:
                mask_pos = random.sample(range(len(sent_list)-1), math.ceil(len(sent_list)*mr))
            except:
                continue
            sent = [token if i not in mask_pos else '_' for i, token in enumerate(sent_list)]
            sents.append(' '.join(sent))
    
    with open(save_path, 'w') as w:
        w.write('\n'.join(sents))


def main():
    dataprocess('./data/PennTreebank/ptb.test.txt', './data/test.txt')


if __name__ == "__main__":
    main()