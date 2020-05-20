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
import random
import logging
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertConfig, BertForTokenClassification, BertForMaskedLM, AdamW
from torch import nn

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

"""
? 定义评估函数
"""
def flat_accuracy(preds, labels):
    preds = preds.flatten()
    labels = labels.flatten()
    return np.sum(preds == labels) / len(labels)

def flat_precision(preds, labels):
    preds = preds.flatten()
    labels = labels.flatten()
    num_tp = np.sum((preds == labels) & (preds == 1))
    num_tp_fp = np.sum(preds == 1)
    
    return 0 if (num_tp_fp == 0) else num_tp / num_tp_fp

def flat_recall(preds, labels):
    preds = preds.flatten()
    labels = labels.flatten()
    num_tp = np.sum((preds == labels) & (preds == 1))
    num_tp_fn = np.sum(labels == 1)
    return 0 if (num_tp_fn == 0) else num_tp / num_tp_fn

def flat_f1(precision, recall):
    mole = 2 * precision * recall
    deno = precision + recall
    return 0 if (deno == 0) else (mole / deno)

def main():
    """
    ? 1. 设置数据
    """
    raw_trifox = []
    with open('../data/TroFi-X/TroFi-X_formatted_svo.csv', encoding='latin-1') as f:
        lines = csv.reader(f)
        next(lines)
        for line in lines:
            sen = line[3].split(' ')
            meta_token_pos = sen.index(line[1])
            sen[meta_token_pos] = '[MASK]'
            label_seq = [0] * len(sen)
            label_seq[meta_token_pos] = int(line[5])
            raw_trifox.append([line[3], label_seq, meta_token_pos, ' '.join(sen)])

    # ! 划分数据集 - 训练集 / 测试集
    random.shuffle(raw_trifox)

    raw_train_trofix, raw_val_trofix = train_test_split(raw_trifox, test_size=0.2, random_state=r)

    tr_sentences = [r[0] for r in raw_train_trofix]
    val_sentences = [r[0] for r in raw_val_trofix]

    tr_labels = [r[1] for r in raw_train_trofix]
    val_labels = [r[1] for r in raw_val_trofix]

    val_verb = [r[2] for r in raw_val_trofix]
    mask_sen = [r[3] for r in raw_val_trofix]

    """
    ? 2. 设置基本参数
    """
    max_len = 60
    batch_size = 8
    output_dir = './trofix_model'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = os.path.join(os.environ['HOME'], 'model/bert-base-uncased')

    """ 
    ? 3. 数据处理
    ! 不使用 BertTokenizer 进行分词，直接使用预料中分割的数据
    """
    tokenizer = BertTokenizer.from_pretrained(model_dir, do_lower_case=True)

    tr_tokenized_texts = [sent.split(' ') for sent in tr_sentences]
    val_tokenized_texts = [sent.split(' ') for sent in val_sentences]

    # ! 对输入进行 encode 和长度固定（截长补短） 
    tr_input_ids = torch.tensor(pad_sequences([tokenizer.encode(txt, add_special_tokens=True) for txt in tr_tokenized_texts],
                              maxlen=max_len, dtype="long", truncating="post", padding="post"))
    val_input_ids = torch.tensor(pad_sequences([tokenizer.encode(txt, add_special_tokens=True) for txt in val_tokenized_texts],
                              maxlen=max_len, dtype="long", truncating="post", padding="post"))

    tr_labels = torch.tensor(pad_sequences([lab for lab in tr_labels],
                            maxlen=max_len, value=0, padding="post", dtype="long", truncating="post"))
    val_labels = torch.tensor(pad_sequences([lab for lab in val_labels],
                            maxlen=max_len,value=0, padding="post", dtype="long", truncating="post"))

    # ! 设置 mask_attention
    tr_masks = torch.tensor([[float(i>0) for i in input_ids] for input_ids in tr_input_ids])
    val_masks = torch.tensor([[float(i>0) for i in input_ids] for input_ids in val_input_ids])

    # ! 定义dataloader,在训练阶段shuffle数据，预测阶段不需要shuffle
    # TensorDataset:https://blog.csdn.net/l770796776/article/details/81261981
    train_data = TensorDataset(tr_input_ids, tr_masks, tr_labels)
    train_sampler = RandomSampler(train_data) # 随机采样
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    val_data = TensorDataset(val_input_ids, val_masks, val_labels)
    val_sampler = SequentialSampler(val_data) # 顺序采样
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=1)

    """
    ? 4. 模型训练
    """
    config = BertConfig.from_pretrained(os.path.join(model_dir, 'config.json'))
    config.output_hidden_states = True
    model = BertForTokenClassification.from_pretrained(model_dir, config=config)
    masked_model = BertForMaskedLM.from_pretrained(model_dir, config=config)

    model.to(device)
    masked_model.to(device)

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

    val_f1s, val_ps, val_rs, val_accs = [], [], [], []

    for epoch in range(num_epochs):
        print('===== Start training: epoch {} ====='.format(epoch + 1))

        model.train()
        tr_loss = 0
        nb_tr_steps = 0

        # ! training
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            outputs = model(b_input_ids, token_type_ids=None,
                         attention_mask=b_input_mask, labels=b_labels)

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

        # ! Validation
        model.eval()
        eval_loss, eval_accuracy, eval_precision, eval_recall, eval_f1 = 0, 0, 0, 0, 0
        nb_eval_steps = 0

        preds, labels = [], []
        for step, batch in enumerate(val_dataloader):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            # masked_input_ids = torch.tensor(tokenizer.encode(mask_sen[step].split())).unsqueeze(0)
            masked_input_ids = torch.tensor(pad_sequences(torch.tensor(tokenizer.encode(mask_sen[step].split())).unsqueeze(0),
                              maxlen=max_len, dtype="long", truncating="post", padding="post"))
            input_mask = torch.tensor([float(i>0) for i in masked_input_ids[0]]).unsqueeze(0)

            masked_input_ids = masked_input_ids.to(device)
            input_mask = input_mask.to(device)

            with torch.no_grad():
                outputs = model(b_input_ids, token_type_ids=None,
                                      attention_mask=b_input_mask, labels=b_labels)
                masked_outputs = masked_model(masked_input_ids, masked_lm_labels=masked_input_ids, attention_mask=input_mask)

            tmp_eval_loss, logits, hidden_states = outputs[:3]
            masked_loss, prediction_scores, masked_hidden_states = masked_outputs[:3]

            verb_states = hidden_states[-1][:, val_verb[step]]
            masked_verb_states = masked_hidden_states[-1][:, val_verb[step]]
            masked_verb_states -= verb_states
            
            splice_states = torch.cat((verb_states, masked_verb_states), dim=-1)

            # ! 定义全连接层
            connected_layer = nn.Linear(in_features = 1536, out_features = 2)
            connected_layer.to(device)

            logits = connected_layer(splice_states)
            
            values, logits = torch.max(F.softmax(logits, dim=-1), dim=-1)[:2]

            verb_logits = logits
            ture_labels = b_labels[:, val_verb[step]]

            # ! detach的方法，将variable参数从网络中隔离开，不参与参数更新
            verb_logits = verb_logits.detach().cpu().numpy()
            ture_labels = ture_labels.cpu().numpy()

            preds.append(verb_logits)
            labels.append(ture_labels)
            nb_eval_steps += 1
            
            eval_loss += tmp_eval_loss.mean().item()

        # ! 计算评估值
        preds = np.array(preds)
        labels = np.array(labels)

        eval_accuracy = flat_accuracy(preds, labels)
        eval_precision = flat_precision(preds, labels)
        eval_recall = flat_recall(preds, labels)
        eval_f1 = flat_f1(eval_precision, eval_recall)
        
        val_accs.append(eval_accuracy)
        val_ps.append(eval_precision)
        val_rs.append(eval_recall)
        val_f1s.append(eval_f1)

        # 打印信息
        print("{:15}{:<.3f}".format('val loss:', eval_loss/nb_eval_steps))
        print("{:15}{:<.3f}".format('val accuracy:', val_accs[epoch]))
        print("{:15}{:<.3f}".format('val precision:', val_ps[epoch]))
        print("{:15}{:<.3f}".format('val recall:', val_rs[epoch]))
        print("{:15}{:<.3f}".format('val f1', val_f1s[epoch]))

        # if (num_epochs % 5) == 0:
        #     print('saving model for epoch {}'.format(epoch + 1))
        #     if not os.path.exists(output_dir + '/model_epoch{}'.format(epoch + 1)):
        #         os.mkdir(output_dir + 'model_epoch{}'.format(epoch + 1))
        #     model_to_save = model.module if hasattr(model, 'module') else model
        #     model_to_save.save_pretrained(output_dir + 'model_epoch{}'.format(epoch + 1))


    print("===== Train Finished =====\n")
    print("{:15}{:<.3f}".format("ave accuracy", sum(val_accs) / num_epochs))
    print("{:15}{:<.3f}".format("ave precision", sum(val_ps) / num_epochs))
    print("{:15}{:<.3f}".format("ave recall", sum(val_rs) / num_epochs))
    print("{:15}{:<.3f}".format("ave f1", sum(val_f1s) / num_epochs))
   
    # ! 保存模型
    if not os.path.exists(output_dir +'/final_model'):
        os.mkdir(output_dir + 'final_model')
    model_to_save = model.module if hasattr(model,'module') else model
    model_to_save.save_pretrained(output_dir +'final_model')

if __name__ == "__main__":
    main()