#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   data_analysis.py
@Time    :   2020/03/30 19:22:21
@Author  :   Quxiansen 
@Version :   1.0
@Contact :   2191002033@cnu.edu.cn
@License :   
@Desc    :   对mask-lm数据进行分析
'''

# here put the import lib
import math
import os
import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type=str, required=True, help='选择语料')
    args = parser.parse_args()

    base_dir = './masklm/'

    if args.corpus == 'vua':
        corpus = os.path.join(base_dir, 'vua.out')
    elif args.corpus == 'trofix':
        corpus = os.path.join(base_dir, 'trofix.out')
    elif args.corpus == 'trofi':
        corpus = os.path.join(base_dir, 'trofi.out')

    meta_indexs = []
    nonmeta_indexs = []
    with open(corpus, 'r+', encoding='utf-8') as f:
        lines = f.readlines()
        count = [int(c.split(':')[1]) for c in lines[2].split(',')]

        for line in lines[4:4 + count[0]]:
            index = line.split(',')[1]
            meta_indexs.append(int(index))

        for line in lines[4 + count[0]:]:
            index = line.split(',')[1]
            nonmeta_indexs.append(int(index))

    meta_indexs.sort()
    nonmeta_indexs.sort()

    # meta_slice = math.ceil(meta_indexs[len(meta_indexs) - 1] / 1000)
    # meta_slice_list = [i*1000 - 1 for i in range(meta_slice + 1)]
    # nonmeta_slice = math.ceil(nonmeta_indexs[len(nonmeta_indexs) - 1] / 1000)
    # nonmeta_slice_list = [i*1000 - 1 for i in range(nonmeta_slice + 1)]

    meta_slice_list = [i * 100 -1 for i in range(11)]
    nonmeta_slice_list = [i * 100 -1 for i in range(11)]

    meta_poss = [0]
    nonmeta_poss = [0]

    for i, mask in enumerate(meta_indexs):
        if mask > 998:
            continue
        while mask > meta_slice_list[0]:
            meta_poss.append(i)
            del meta_slice_list[0]

    for i, mask in enumerate(nonmeta_indexs):
        if mask > 998:
            continue
        while mask > nonmeta_slice_list[0]:
            nonmeta_poss.append(i)
            del nonmeta_slice_list[0]
    
    meta_poss.append(len(meta_indexs))
    nonmeta_poss.append(len(nonmeta_indexs))

    meta_lengths = [meta_poss[i+1] - meta_poss[i] for i in range(len(meta_poss) - 1)]
    nonmeta_lengths = [nonmeta_poss[i+1] - nonmeta_poss[i] for i in range(len(nonmeta_poss) - 1)]

    print('隐喻词分布：')
    print('{:13}: {}'.format(-1, meta_lengths[0]))
    for i, length in enumerate(meta_lengths):
        if i == 0:
            continue
        print('{:5} ~ {:5}: {}'.format((i-1)*100, i*100 - 1, length))

    print('平均排名：{}'.format(sum(meta_indexs) / len(meta_indexs)))

    print('\n非隐喻词分布：')
    print('{:13}: {}'.format(-1, nonmeta_lengths[0]))
    for i, length in enumerate(nonmeta_lengths):
        if i == 0:
            continue
        print('{:5} ~ {:5}: {}'.format((i-1)*100, i*100 - 1, length))

    print('平均排名：{}'.format(sum(nonmeta_indexs) / len(nonmeta_indexs)))


if __name__ == "__main__":
    main()