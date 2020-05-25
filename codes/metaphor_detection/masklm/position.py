#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   position.py
@Time    :   2020/05/22 18:10:53
@Author  :   Aiken 
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


def predicte(raw_data, model_dir):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    """ 加载模型 """
    config = BertConfig.from_pretrained(model_dir + '/config.json')
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForMaskedLM.from_pretrained(model_dir, config=config)

    model.to(device)

    """ 预测 """
    pred, nin_pred = [], []
    for step, data in enumerate(raw_data):
        is_meta, verb, sent, m_sent, verb_idx = data
        sent = sent.split(' ')

        input_ids = torch.tensor(tokenizer.encode(m_sent, add_special_tokens=True)).unsqueeze(0)

        for i, id in enumerate(input_ids[0]):
            if id == 103:
                mask_index = i

        # 切换到 gpu 上运行
        input_ids = input_ids.to(device)

        with torch.no_grad():
            outputs = model(input_ids)
        
        prediction_scores = outputs[0]

        # 对预测后的分数做 softmax 取前5个最大值
        sm = F.softmax(prediction_scores, dim=2)
        # sort_values, sort_indices = sm.topk(5, dim=2)[:2]
        sort_values, sort_indices = torch.sort(sm, dim=2, descending=True)[:2]

        # 取出预测词 values 和 indices
        values = (sort_values[0][mask_index]).tolist()
        indices = (sort_indices[0][mask_index]).tolist()

        # 将预测词 decode
        for i, indice in enumerate(indices):
            indices[i] = tokenizer.decode(indice).replace(' ', '')

        try:
            index = indices.index(verb)
        except:
            index = 'NIN'

        idx_pred = [str(index)]
        idx_pred.extend(indices[:5])
        
        sent[verb_idx] = verb + '_[' + ', '.join(idx_pred) + ']'

        temp = is_meta + '\t' + ' '.join(sent)
        print(temp)

        if index == 'NIN':
            nin_pred.append(temp)
        else:
            pred.append(temp)
        
    return pred, nin_pred


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type=str, required=True, help='选择语料')
    parser.add_argument('--model_dir', default='bert-base-uncased', type=str, required=True, help='模型目录')
    parser.add_argument('--device', default='0', type=str, required=False, help='生成设备')
    parser.add_argument('--save', action='store_true', help='是否保存结果')
    parser.add_argument('--output_dir', default='./pred.txt', type=str, required=False, help='结果保存目录')

    args = parser.parse_args()
    print('args:\n' + args.__repr__())

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    """ 读取数据 """
    raw_data = []

    if args.corpus == 'vua':
        pass
    elif args.corpus == 'trofix':
        with open('../data/TroFi-X/TroFi-X_formatted_svo.csv',  encoding='latin-1') as f:
            lines = csv.reader(f)
            next(lines)
            for line in lines:
                is_meta = 'Y' if int(line[5]) == 1 else 'N'
                sen = line[3].split(' ')
                verb_idx = sen.index(line[1])
                sen[verb_idx] = '[MASK]'
                raw_data.append([is_meta, line[1], line[3], ' '.join(sen), verb_idx])

    elif args.corpus == 'trofi':
        with open('../data/TroFi/TroFi_formatted_all3737.csv',  encoding='latin-1') as f:
            lines = csv.reader(f)
            next(lines)
            for line in lines:
                is_meta = 'Y' if int(line[3]) == 1 else 'N'
                sen = line[1].split(' ')
                verb = sen[int(line[2])]
                sen[int(line[2])] = '[MASK]'
                raw_data.append([is_meta, verb, line[1], ' '.join(sen), int(line[2])])

    pred, nin_pred = predicte(raw_data, args.model_dir)

    if args.save:
        print('写入文件中...')
        with open(args.output_dir, 'w') as writer:
            writer.write('\n'.join(pred))
        
        with open('./ninpred.txt', 'w') as writer:
            writer.write('\n'.join(nin_pred))



if __name__ == "__main__":
    main()