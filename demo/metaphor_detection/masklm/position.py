#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   mlm.py
@Time    :   2020/04/16 21:31:00
@Author  :   Quxiansen 
@Version :   1.0
@Contact :   2191002033@cnu.edu.cn
@License :   
@Desc    :   None
'''

# here put the import lib

import os
import csv
import ast
import random
import argparse
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertConfig


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', type=str, required=False, help='生成设备')
    parser.add_argument('--corpus', type=str, required=True, help='选择语料')

    args = parser.parse_args()
    print('args:\n' + args.__repr__())

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    """ 加载模型 """

    model_dir = os.path.join(os.environ['HOME'], 'sources/bert-base-uncased')
    config = BertConfig.from_pretrained(model_dir + '/config.json')
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForMaskedLM.from_pretrained(model_dir, config=config)

    model.to(device)

    """ 读取数据 """
    raw_meta_data = []
    raw_nonmeta_data = []

    if args.corpus == 'vua':
        with open('../data/VUA/VUA_formatted_train.csv', encoding='latin-1') as f:
            lines = csv.reader(f)
            next(lines)
            for line in lines:
                sen = line[3].split(' ')
                verb = sen[int(line[4])]
                sen[int(line[4])] = '[MASK]'
                if line[5] == '1':
                    raw_meta_data.append([verb, line[3], ' '.join(sen)])
                else:
                    raw_nonmeta_data.append([verb, line[3], ' '.join(sen)])

    elif args.corpus == 'trofix':
        with open('../data/TroFi-X/TroFi-X_formatted_svo.csv',  encoding='latin-1') as f:
            lines = csv.reader(f)
            next(lines)
            for line in lines:
                sen = line[3].split(' ')
                verb = sen.index(line[1])
                sen[verb] = '[MASK]'
                if line[5] == '1':
                    raw_meta_data.append([line[1], line[3], ' '.join(sen)])
                else:
                    raw_nonmeta_data.append([line[1], line[3], ' '.join(sen)])

    elif args.corpus == 'trofi':
        with open('../data/TroFi/TroFi_formatted_all3737.csv',  encoding='latin-1') as f:
            lines = csv.reader(f)
            next(lines)
            for line in lines:
                sen = line[1].split(' ')
                verb = sen[int(line[2])]
                sen[int(line[2])] = '[MASK]'
                if line[3] == '1':
                    raw_meta_data.append([verb, line[1], ' '.join(sen)])
                else:
                    raw_nonmeta_data.append([verb, line[1], ' '.join(sen)])

    meta_poss = []
    nonmeta_poss = []
    """ 隐喻词 """
    for data in raw_meta_data:
        verb = data[0]
        sentence = data[2]

        input_ids = torch.tensor(tokenizer.encode(sentence, add_special_tokens=True)).unsqueeze(0)

        for i, id in enumerate(input_ids[0]):
            if id == 103:
                mask_index = i

        # 切换到 gpu 上运行
        input_ids = input_ids.to(device)

        outputs = model(input_ids)
        prediction_scores = outputs[0]

        # 对预测后的分数做 softmax 取前5个最大值
        sm_result = F.softmax(prediction_scores, dim=2)
        # topk_values, topk_indices = sm_result.topk(5, dim=2)[:2]
        sort_values, sort_indices = torch.sort(sm_result, dim=2, descending=True)[:2]

        # 取出预测词 values 和 indices
        values = (sort_values[0][mask_index]).tolist()
        indices = (sort_indices[0][mask_index]).tolist()

        # 将预测词 decode
        for i, indice in enumerate(indices):
            indices[i] = tokenizer.decode(indice).replace(' ', '')

        try:
            index = indices.index(verb)
        except:
            index = -1

        temp = [verb, str(index)]
        meta_poss.append(','.join(temp))

    """ 非隐喻词 """
    for data in raw_nonmeta_data:
        verb = data[0]
        sentence = data[2]

        input_ids = torch.tensor(tokenizer.encode(sentence, add_special_tokens=True)).unsqueeze(0)

        for i, id in enumerate(input_ids[0]):
            if id == 103:
                mask_index = i

        # 切换到 gpu 上运行
        input_ids = input_ids.to(device)

        outputs = model(input_ids)
        prediction_scores = outputs[0]

        # 对预测后的分数做 softmax 取前5个最大值
        sm_result = F.softmax(prediction_scores, dim=2)
        # topk_values, topk_indices = sm_result.topk(5, dim=2)[:2]
        sort_values, sort_indices = torch.sort(sm_result, dim=2, descending=True)[:2]

        # 取出预测词 values 和 indices
        values = (sort_values[0][mask_index]).tolist()
        indices = (sort_indices[0][mask_index]).tolist()

        # 将预测词 decode
        for i, indice in enumerate(indices):
            indices[i] = tokenizer.decode(indice).replace(' ', '')

        try:
            index = indices.index(verb)
        except:
            index = -1

        temp = [verb, str(index)]
        nonmeta_poss.append(','.join(temp))
    
    print("{} {}".format('word', 'index'))
    print('\n'.join(meta_poss))
    print('共 {} 个隐喻词'.format(len(meta_poss)))
    print('\n'.join(nonmeta_poss))
    print('共 {} 个非隐喻词'.format(len(nonmeta_poss)))


if __name__ == "__main__":
    main()