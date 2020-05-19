#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   models.py
@Time    :   2020/04/21 18:08:21
@Author  :   Aiken 
@Version :   1.0
@Contact :   2191002033@cnu.edu.cn
@License :   
@Desc    :   None
'''

# here put the import lib
import os
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForMaskedLM, BertConfig


# 加载模型
model_dir = os.path.join(os.environ['HOME'], 'model/bert-base-uncased')  # 模型目录
config = BertConfig.from_pretrained(model_dir + '/config.json')
config.output_hidden_states=True
tokenizer = BertTokenizer.from_pretrained(model_dir)
model = BertForMaskedLM.from_pretrained(model_dir, config=config)

sentence = 'They are irate about [MASK] capital-adequacy requirements that force securities firms to pump at least 20 % more capital into reserves .'

input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)

index_list = []

for i, id in enumerate(input_ids[0]):
    if id == 103:
        index_list.append(i)

 # 切换到 gpu 上运行
if torch.cuda.is_available():
    input_ids = input_ids.to('cuda')
    model.to('cuda')

outputs = model(input_ids, masked_lm_labels=input_ids)
loss, prediction_scores, hidden_states = outputs[:3]

 # 对预测后的分数做 softmax 取前5个最大值
sm_result = F.softmax(prediction_scores, dim=2)
topk_values, topk_indices = sm_result.topk(5, dim=2)[:2]

 # 取出预测词 values 和 indices
mask_values = (topk_values[0][index_list]).tolist()
mask_indices = (topk_indices[0][index_list]).tolist()

 # 将预测词 decode
for i, indices in enumerate(mask_indices):
    # mask_indices[i] = tokenizer.decode(indices).split(' ')
    for j, indice in enumerate(indices):
        mask_indices[i][j] = tokenizer.decode(indice).replace(' ', '')
        

print(mask_indices)