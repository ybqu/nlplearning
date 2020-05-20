#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   trofix_bert.py
@Time    :   2020/04/29 23:44:57
@Author  :   Aiken 
@Version :   1.0
@Contact :   2191002033@cnu.edu.cn
@License :   
@Desc    :   使用 Bert 在 VUA 数据集上进行隐喻词识别
'''

# here put the import lib

import os
import csv
import ast
import torch
import random
import logging
import argparse
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertConfig, BertForTokenClassification, AdamW

# os.environ['CUDA_VISIBLE_DEVICES']='1'

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging = logging.getLogger(__name__)


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


def load_vua(train_data_dir, val_data_dir):
    """ 读取 VUA 数据
    :param train_data_dir: 训练数据目录 (../data/VUAsequence/VUA_seq_formatted_train.csv)
    :param val_data_dir: 测试数据目录 (../data/VUAsequence/VUA_seq_formatted_val.csv)
    """
    raw_train_vua = []
    raw_val_vua = []

    with open(train_data_dir, encoding='latin-1') as f:
        lines = csv.reader(f)
        next(lines)
        for line in lines:
            pos_seq = ast.literal_eval(line[4])
            label_seq = ast.literal_eval(line[3])
            assert (len(pos_seq) == len(label_seq))
            assert (len(line[2].split()) == len(pos_seq))
            raw_train_vua.append([line[2], label_seq, pos_seq])

    with open(val_data_dir, encoding='latin-1') as f:
        lines = csv.reader(f)
        next(lines)
        for line in lines:
            pos_seq = ast.literal_eval(line[4])
            label_seq = ast.literal_eval(line[3])
            assert (len(pos_seq) == len(label_seq))
            assert (len(line[2].split()) == len(pos_seq))
            raw_val_vua.append([line[2], label_seq, pos_seq])

    tr_sentences = [r[0] for r in raw_train_vua]
    val_sentences = [r[0] for r in raw_val_vua]

    tr_labels = [r[1] for r in raw_train_vua]
    val_labels = [r[1] for r in raw_val_vua]

    return tr_sentences, tr_labels, val_sentences, val_labels


def load_trofi(train_data_dir, val_data_dir, seed):
    """ 读取 TroFi 数据
    :param train_data_dir: 训练数据目录 (../data/TroFi/TroFi_formatted_all3737.csv)
    :param val_data_dir: 
    :param seed: 划分数据集随机数种子
    """
    raw_trofi = []

    with open(train_data_dir, encoding='latin-1') as f:
        lines = csv.reader(f)
        next(lines)
        for line in lines:
            sen = line[1].split(' ')
            label_seq = [0] * len(sen)
            label_seq[int(line[2])] = int(line[3])
            assert (len(label_seq) == len(sen))
            raw_trofi.append([line[1], label_seq, int(line[2])])
    
    # random.shuffle(raw_trofi)
    raw_train_trofi, raw_val_trofi = train_test_split(raw_trofi, test_size=0.2, random_state=seed)

    tr_sentences = [r[0] for r in raw_train_trofi]
    val_sentences = [r[0] for r in raw_val_trofi]

    tr_labels = [r[1] for r in raw_train_trofi]
    val_labels = [r[1] for r in raw_val_trofi]

    val_verb = [r[2] for r in raw_val_trofi]

    return tr_sentences, tr_labels, val_sentences, val_labels, val_verb


def load_trofix(train_data_dir, val_data_dir, seed):
    """ 读取 TroFi 数据
    :param train_data_dir: 训练数据目录 (../data/TroFi-X/TroFi-X_formatted_svo.csv)
    :param val_data_dir: 
    :param seed: 划分数据集随机数种子
    """
    raw_trofix = []

    with open(train_data_dir, encoding='latin-1') as f:
        lines = csv.reader(f)
        next(lines)
        for line in lines:
            sen = line[3].split(' ')
            label_seq = [0] * len(sen)
            label_seq[sen.index(line[1])] = int(line[5])
            raw_trofix.append([line[3], label_seq, sen.index(line[1])])

    # random.shuffle(raw_trofix)
    raw_train_trofix, raw_val_trofix = train_test_split(raw_trofix, test_size=0.2, random_state=seed)

    tr_sentences = [r[0] for r in raw_train_trofix]
    val_sentences = [r[0] for r in raw_val_trofix]

    tr_labels = [r[1] for r in raw_train_trofix]
    val_labels = [r[1] for r in raw_val_trofix]

    val_verb = [r[2] for r in raw_val_trofix]

    return tr_sentences, tr_labels, val_sentences, val_labels, val_verb


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', type=str,
                        required=False, help='选择设备')
    parser.add_argument('--corpus', type=str, required=True, help='选择语料')
    parser.add_argument('--seed', default=4, type=int,
                        required=False, help='输入种子数')
    parser.add_argument('--model_dir', type=str, required=True, help='模型目录')
    parser.add_argument('--train_data_dir', type=str,
                        required=True, help='训练数据目录')
    parser.add_argument('--val_data_dir', type=str,
                        required=True, help='测试数据目录')
    parser.add_argument('--max_len', default=60, type=int, required=False, help='句子最大长度')
    parser.add_argument('--batch_size', default=8, type=int, required=False, help='训练batch_size')
    parser.add_argument('--lr', default=3e-5, type=float, required=False, help='学习率')
    parser.add_argument('--num_epochs', default=15, type=int, required=False, help='训练epoch')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--save', action='store_true', help='是否保存模型')
    parser.add_argument('--output_dir', default='./model/', type=str, required=False, help='模型输出路径')
    args = parser.parse_args()

    print('args:\n' + args.__repr__())

    # ? 种子数设置
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # ! 用以保证实验的可重复性，使每次运行的结果完全一致
    torch.backends.cudnn.deterministic = True

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = BertConfig.from_pretrained(args.model_dir + '/config.json')
    tokenizer = BertTokenizer.from_pretrained(args.model_dir, do_lower_case=True)
    model = BertForTokenClassification.from_pretrained(
        args.model_dir, config=config)

    model.to(device)

    """
    ? 1. 设置数据
    """
    if args.corpus == 'vua':
        tr_sentences, tr_labels, val_sentences, val_labels = load_vua(
            args.train_data_dir, args.val_data_dir)
    elif args.corpus == 'trofi':
        tr_sentences, tr_labels, val_sentences, val_labels, val_verb = load_trofi(
            args.train_data_dir, args.val_data_dir, args.seed)
    elif args.corpus == 'trofix':
        tr_sentences, tr_labels, val_sentences, val_labels, val_verb = load_trofix(
            args.train_data_dir, args.val_data_dir, args.seed)

    """ 
    ? 3. 数据处理
    ! 不使用 BertTokenizer 进行分词，直接使用预料中分割的数据
    """
    tr_tokenized_texts = [sent.split(' ') for sent in tr_sentences]
    val_tokenized_texts = [sent.split(' ') for sent in val_sentences]

    # ! 对输入进行 encode 和长度固定（截长补短）
    tr_input_ids = torch.tensor(pad_sequences([tokenizer.encode(txt) for txt in tr_tokenized_texts],
                                              maxlen=args.max_len, dtype="long", truncating="post", padding="post"))
    val_input_ids = torch.tensor(pad_sequences([tokenizer.encode(txt) for txt in val_tokenized_texts],
                                               maxlen=args.max_len, dtype="long", truncating="post", padding="post"))

    tr_labels = torch.tensor(pad_sequences([lab for lab in tr_labels],
                                           maxlen=args.max_len, value=0, padding="post", dtype="long", truncating="post"))
    val_labels = torch.tensor(pad_sequences([lab for lab in val_labels],
                                            maxlen=args.max_len, value=0, padding="post", dtype="long", truncating="post"))

    # ! 设置 mask_attention
    tr_masks = torch.tensor([[float(i > 0) for i in input_ids]
                             for input_ids in tr_input_ids])
    val_masks = torch.tensor([[float(i > 0) for i in input_ids]
                              for input_ids in val_input_ids])

    # ! 定义dataloader,在训练阶段shuffle数据，预测阶段不需要shuffle
    # TensorDataset:https://blog.csdn.net/l770796776/article/details/81261981
    train_data = TensorDataset(tr_input_ids, tr_masks, tr_labels)
    train_sampler = RandomSampler(train_data)  # 随机采样
    train_dataloader = DataLoader(
        train_data, sampler=train_sampler, batch_size=args.batch_size)

    val_data = TensorDataset(val_input_ids, val_masks, val_labels)
    val_sampler = SequentialSampler(val_data)  # 顺序采样
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=1)

    """
    ? 4. 模型训练
    """
    # ! 定义 optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {"params": [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)

    """ 
    ? 5. 开始微调
    """
    num_epochs = args.num_epochs

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
            torch.nn.utils.clip_grad_norm_(
                parameters=model.parameters(), max_norm=args.max_grad_norm)
            # ! 更新参数
            optimizer.step()
            model.zero_grad()

        print("\nEpoch {} of training loss: {}".format(
            epoch + 1, tr_loss/nb_tr_steps))

        # ! Validation
        model.eval()
        eval_loss, eval_accuracy, eval_precision, eval_recall, eval_f1 = 0, 0, 0, 0, 0
        nb_eval_steps = 0

        preds, labels = [], []
        for step, batch in enumerate(val_dataloader):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            with torch.no_grad():
                outputs = model(b_input_ids, token_type_ids=None,
                                attention_mask=b_input_mask, labels=b_labels)

            tmp_eval_loss, logits = outputs[:2]

            values, logits = torch.max(F.softmax(logits, dim=-1), dim=-1)[:2]

            ture_labels = b_labels

            if args.corpus == 'trofi' or args.corpus == 'trofix':
                logits = logits[0][val_verb[step]]
                ture_labels = ture_labels[0][val_verb[step]]

            # ! detach的方法，将variable参数从网络中隔离开，不参与参数更新
            logits = logits.detach().cpu().numpy()
            ture_labels = ture_labels.cpu().numpy()

            preds.append(logits)
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

        if (num_epochs % 5) == 0:
            print('saving model for epoch {}'.format(epoch + 1))
            if not os.path.exists(args.output_dir + 'model_epoch{}'.format(epoch + 1)):
                os.mkdir(args.output_dir + 'model_epoch{}'.format(epoch + 1))
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(args.output_dir + 'model_epoch{}'.format(epoch + 1))

    print("===== Train Finished =====\n")
    print("{:15}{:<.3f}".format("ave accuracy", sum(val_accs) / num_epochs))
    print("{:15}{:<.3f}".format("ave precision", sum(val_ps) / num_epochs))
    print("{:15}{:<.3f}".format("ave recall", sum(val_rs) / num_epochs))
    print("{:15}{:<.3f}".format("ave f1", sum(val_f1s) / num_epochs))

    # ! 保存模型
    if args.save:
        if not os.path.exists(args.output_dir + 'final_model'):
            os.mkdir(args.output_dir + 'final_model')
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(args.output_dir + 'final_model')


if __name__ == "__main__":
    main()
