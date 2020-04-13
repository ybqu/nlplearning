#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   bertformaskedlm.py
@Time    :   2019/11/21 13:29:51
@Author  :   Qu Yuanbin 
@Version :   1.0
@Contact :   2191002033@cnu.edu.cn
@License :   
@Desc    :   使用 BertForMaskedLM 生成文本
'''

# here put the import lib
import random
import os
import csv
import ast
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertConfig

os.environ['CUDA_VISIBLE_DEVICES']='1'

# 加载模型
ROOT_PATH = os.environ['HOME']
model_dir = os.path.join(ROOT_PATH, 'sources/bert-base-uncased')
config = BertConfig.from_pretrained(model_dir + '/config.json')
tokenizer = BertTokenizer.from_pretrained(model_dir)
model = BertForMaskedLM.from_pretrained(model_dir, config=config)

""" 读取数据 """
raw_analysis_vua = []
with open('../../analysis/vua_masked.csv', encoding='latin-1') as f:
    lines = csv.reader(f)
    # next(lines)
    for line in lines:
        label_seq = ast.literal_eval(line[1])
        raw_analysis_vua.append([line[0], label_seq, line[2]])

pos_list = []
for k, vua in enumerate(raw_analysis_vua):
    # sentence = input('sentence:')
    print('\n============= SENTENCE ' + str(k + 1) + ' =============')
    sentence = vua[0]
    m_sentence = vua[2]
    sen_list = sentence.split(' ')
    m_sen_list = m_sentence.split(' ')
    m_token = [sen_list[i] for i, token in enumerate(m_sen_list) if token == '[MASK]']
    
    print('ORIGINAL >> ' + sentence)
    print('MASKED   >> ' + m_sentence)

    input_ids = torch.tensor(tokenizer.encode(m_sentence, add_special_tokens=True)).unsqueeze(0)

    index_list = []

    for i, id in enumerate(input_ids[0]):
        if id == 103:
            index_list.append(i)

    # 切换到 gpu 上运行
    if torch.cuda.is_available():
        input_ids = input_ids.to('cuda')
        model.to('cuda')

    outputs = model(input_ids, masked_lm_labels=input_ids)
    loss, prediction_scores = outputs[:2]

    # 对预测后的分数做 softmax 取前5个最大值
    sm_result = F.softmax(prediction_scores, dim=2)
    # topk_values, topk_indices = sm_result.topk(5, dim=2)[:2]
    sort_values, sort_indices = torch.sort(sm_result, dim=2, descending=True)[:2]

    # 取出预测词 values 和 indices
    mask_values = (sort_values[0][index_list]).tolist()
    mask_indices = (sort_indices[0][index_list]).tolist()

    # 将预测词 decode
    for i, indices_list in enumerate(mask_indices):
        for j, indices in enumerate(indices_list):
            mask_indices[i][j] = tokenizer.decode(indices).replace(' ', '')
    
    print('\n预测：')
    print('{:12} {:5} {} \n'.format('MASKED_TOKEN', 'INDEX', 'TOP30'))
    for i, indices in enumerate(mask_indices):
        try:
            index = indices.index(m_token[i])
        except:
            index = -1
        temp = [m_token[i], index]
        pos_list.append(temp)
        print('{:12} {:5} {}'.format(m_token[i], index, ','.join(indices[:30])))

print('\n' + '*'*100)
print('([MASK] INDEX) 共 {} 条数据：\n'.format(len(pos_list)))
for i, info in enumerate(pos_list):
    print(' '.join(info))