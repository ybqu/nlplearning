#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   trofix_bert.py
@Time    :   2020/04/29 23:44:57
@Author  :   Aiken 
@Version :   1.0
@Contact :   2191002033@cnu.edu.cn
@License :   
@Desc    :   使 masked-lm 和 预测做拼接
'''

# here put the import lib

import os
import csv
import torch
import logging
import torch.nn.functional as F
from transformers import BertTokenizer, BertConfig, BertForTokenClassification, AdamW

# os.environ['CUDA_VISIBLE_DEVICES']='1'

# logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logging = logging.getLogger(__name__)

def main():
    """
    ? 1. 设置数据
    """
    raw_data = []
    # with open('../data/TroFi-X/TroFi-X_formatted_svo.csv', encoding='latin-1') as f:
    #     lines = csv.reader(f)
    #     next(lines)
    #     for line in lines:
    #         sent = line[3].split(' ')
    #         raw_data.append([line[3], line[1], sent.index(line[1]), int(line[5])])
    with open('../data/TroFi/TroFi_formatted_all3737.csv', encoding='latin-1') as f:
        lines = csv.reader(f)
        next(lines)
        for line in lines:
            raw_data.append([line[1], line[0], line[2], int(line[3])])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_dir = './model'

    tokenizer = BertTokenizer.from_pretrained(model_dir, do_lower_case=True)
    config = BertConfig.from_pretrained(os.path.join(model_dir, 'config.json'))
    model = BertForTokenClassification.from_pretrained(model_dir, config=config)

    model.to(device)

    tp, tn, fp, fn = [], [], [], []

    for step, batch in enumerate(raw_data):
        sent, verb, pos, label = batch

        input_ids = torch.tensor(tokenizer.encode(sent.split(' '), add_special_tokens=True)).unsqueeze(0)  # Batch size 1

        input_ids = input_ids.to(device)

        with torch.no_grad():
            outputs = model(input_ids)
        
        logits = outputs[0][:, pos + 1]

        values, logits = torch.max(F.softmax(logits, dim=-1), dim=-1)[:2]

        # print(logits.item())
        temp = [sent, verb, str(label), str(logits.item())]
        temp = ','.join(temp)
        if label == 1 and logits.item() == 1:
            tp.append(temp)
        elif label == 1 and logits.item() == 0:
            fn.append(temp)
        elif label == 0 and logits.item() == 0:
            tn.append(temp)
        elif label == 0 and logits.item() == 1:
            fp.append(temp)

    print('sentence, verb, ture_label, prediction_label')
    print('='*10 + ' TP(1 -> 1) ' + '='*20 + ' 共' + str(len(tp)) + '条数据')
    print('\n'.join(tp))
    print('='*10 + ' TN(0 -> 0) ' + '='*20 + ' 共' + str(len(tn)) + '条数据')
    print('\n'.join(tn))
    print('='*10 + ' FP(0 -> 1) ' + '='*20 + ' 共' + str(len(fp)) + '条数据')
    print('\n'.join(fp))
    print('='*10 + ' FN(1 -> 0) ' + '='*20 + ' 共' + str(len(fn)) + '条数据')
    print('\n'.join(fn))


if __name__ == "__main__":
    main()