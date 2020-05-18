#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   trofix_bert.py
@Time    :   2020/04/29 23:44:57
@Author  :   Aiken 
@Version :   1.0
@Contact :   2191002033@cnu.edu.cn
@License :   
@Desc    :   Modified based on Gao Ge https://github.com/gao-g/metaphor-in-context.
             trofix corpus
'''

# here put the import lib

import os
import csv
import torch
import random
import logging
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertConfig, BertForMaskedLM, AdamW

# os.environ['CUDA_VISIBLE_DEVICES']='1'

logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging = logging.getLogger(__name__)

# ? 种子数设置
r=4
random.seed(r)
np.random.seed(r)
torch.manual_seed(r)
torch.cuda.manual_seed(r)
# ! 用以保证实验的可重复性，使每次运行的结果完全一致
torch.backends.cudnn.deterministic = True

def main():
    """
    ? 1. 设置数据
    """
    raw_trofix = []
    with open('../metaphor_detection/data/TroFi-X/TroFi-X_formatted_svo.csv', encoding='latin-1') as f:
        lines = csv.reader(f)
        next(lines)
        for line in lines:
            sen = line[3].split(' ')
            label_seq = [0] * len(sen)
            label_seq[sen.index(line[1])] = int(line[5])
            raw_trofix.append([line[3], label_seq, sen.index(line[1])])

    tr_sentences = [r[0] for r in raw_trofix]
    # val_sentences = [r[0] for r in raw_trofix]

    """
    ? 2. 设置基本参数
    """
    output_dir = './trofix_g_model'


    # tr_tokenized_texts = [sent.split(' ') for sent in tr_sentences]

    # tr_input_ids = torch.tensor([tokenizer.encode(txt) for txt in tr_tokenized_texts])

    """
    ? 4. 模型训练
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_dir = os.path.join(os.environ['HOME'], 'model/bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained(model_dir, do_lower_case=True)
    config = BertConfig.from_pretrained(os.path.join(model_dir, 'config.json'))
    model = BertForMaskedLM.from_pretrained(model_dir, config=config)

    model.to(device)

    # ! 定义 optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5)

    """ 
    ? 5. 开始微调
    """
    num_epochs = 15
    max_grad_norm = 1.0

    for epoch in range(num_epochs):
        logging.info('Start training: epoch {}'.format(epoch + 1))

        model.train()
        tr_loss = 0
        nb_tr_steps = 0

        # ! training
        for step, sent in enumerate(tr_sentences):
            input_ids = torch.tensor(tokenizer.encode(sent.split(' '), add_special_tokens=True)).unsqueeze(0)
            input_ids = input_ids.to(device)

            outputs = model(input_ids, masked_lm_labels=input_ids)

            loss = outputs[0]
            loss.backward()

            tr_loss += float(loss.item())

            nb_tr_steps += 1

            # ! 减小梯度 https://www.cnblogs.com/lindaxin/p/7998196.html
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
            # ! 更新参数
            optimizer.step()
            model.zero_grad()

        print("\nEpoch {} of training loss: {}".format(epoch + 1, tr_loss/nb_tr_steps))

        model.eval()
        
    logging.info('Training finished')

    # ! 保存模型
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(output_dir)

if __name__ == "__main__":
    main()