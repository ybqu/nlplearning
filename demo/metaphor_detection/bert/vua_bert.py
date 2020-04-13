import torch
import torch.nn as nn
import torch.optim as optim
import csv
import ast
import random
import os
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertConfig, BertForTokenClassification, AdamW

# os.environ['CUDA_VISIBLE_DEVICES']='1'

# Modified based on Gao Ge https://github.com/gao-g/metaphor-in-context
print("PyTorch version:")
print(torch.__version__)
print("GPU Detected:")
print(torch.cuda.is_available())

# 种子数设置
r=4
random.seed(r)
np.random.seed(r)
torch.manual_seed(r)
torch.cuda.manual_seed(r)
# 用以保证实验的可重复性，使每次运行的结果完全一致
torch.backends.cudnn.deterministic = True

"""
1. 读取数据
"""
pos_set = set()
raw_train_vua = [] # [[sen_ix, label_seq, pos_seq], ..., ]
raw_val_vua = []

with open('../data/VUAsequence/VUA_seq_formatted_train.csv', encoding='latin-1') as f:
    lines = csv.reader(f)
    next(lines)
    for line in lines:
        pos_seq = ast.literal_eval(line[4])
        label_seq = ast.literal_eval(line[3])
        assert (len(pos_seq) == len(label_seq))
        assert (len(line[2].split()) == len(pos_seq))
        raw_train_vua.append([line[2], label_seq, pos_seq])
        pos_set.update(pos_seq)

with open('../data/VUAsequence/VUA_seq_formatted_val.csv', encoding='latin-1') as f:
    lines = csv.reader(f)
    next(lines)
    for line in lines:
        pos_seq = ast.literal_eval(line[4])
        label_seq = ast.literal_eval(line[3])
        assert (len(pos_seq) == len(label_seq))
        assert (len(line[2].split()) == len(pos_seq))
        raw_val_vua.append([line[2], label_seq, pos_seq])
        pos_set.update(pos_seq)

# sen_ix, label_seq, pos_seq
# TODO pos_seq 是否要用于训练
# FIXME 
tr_sentences = [r[0] for r in raw_train_vua]
val_sentences = [r[0] for r in raw_val_vua]

tr_labels = [r[1] for r in raw_train_vua]
val_labels = [r[1] for r in raw_val_vua]

tr_pos = [r[2] for r in raw_train_vua]
val_pos = [r[2] for r in raw_val_vua]

"""
2. 设置基本参数
"""
max_len = 60
batch_size = 8
output_dir = './vua_model'

"""
3. 设置device
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
torch.cuda.get_device_name(0)

""" 
4. tokenize 处理
    + 不使用 BertTokenizer 进行分词，直接使用预料中分割的数据
"""
tokenizer = BertTokenizer.from_pretrained('/home/ybqu/sources/bert-base-uncased', do_lower_case=True)
# tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

tr_tokenized_texts = [sent.split(' ') for sent in tr_sentences]
val_tokenized_texts = [sent.split(' ') for sent in val_sentences]

""" 
5. 输入转换为id，
    + 截长补短；
    + 长度为 60；
"""
# input_ids = [tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts]
# pad_sequences: https://blog.csdn.net/wcy23580/article/details/84957471
tr_input_ids = torch.tensor(pad_sequences([tokenizer.encode(txt) for txt in tr_tokenized_texts],
                          maxlen=max_len, dtype="long", truncating="post", padding="post"))
val_input_ids = torch.tensor(pad_sequences([tokenizer.encode(txt) for txt in val_tokenized_texts],
                          maxlen=max_len, dtype="long", truncating="post", padding="post"))


tr_labels = torch.tensor(pad_sequences([lab for lab in tr_labels],
                        maxlen=max_len, value=0, padding="post", dtype="long", truncating="post"))

val_labels = torch.tensor(pad_sequences([lab for lab in val_labels],
                        maxlen=max_len,value=0, padding="post", dtype="long", truncating="post"))
""" 
6. 准备mask_attention
"""
tr_masks = torch.tensor([[float(i>0) for i in ii] for ii in tr_input_ids])
val_masks = torch.tensor([[float(i>0) for i in ii] for ii in val_input_ids])

""" 
7. 定义dataloader,在训练阶段shuffle数据，预测阶段不需要shuffle
"""
# TensorDataset:https://blog.csdn.net/l770796776/article/details/81261981
train_data = TensorDataset(tr_input_ids, tr_masks, tr_labels)
train_sampler = RandomSampler(train_data) # 随机采样
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

valid_data = TensorDataset(val_input_ids, val_masks, val_labels)
valid_sampler = SequentialSampler(valid_data) # 顺序采样
valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=batch_size)

"""
8. 模型训练
"""
config = BertConfig.from_pretrained('/home/ybqu/sources/bert-base-uncased/config.json')
model = BertForTokenClassification.from_pretrained('/home/ybqu/sources/bert-base-uncased', config=config)

# Move the model to the GPU if available
if torch.cuda.is_available():
    model.to('cuda')

"""
9. 定义评估函数 && optimizer

f1: https://blog.csdn.net/qq_37466121/article/details/87719044
"""
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=2).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def flat_precision(preds, labels):
    pred_flat = np.argmax(preds, axis=2).flatten()
    labels_flat = labels.flatten()
    num_tp = np.sum((pred_flat == labels_flat) & (pred_flat == 1))
    num_tp_fp = np.sum(pred_flat == 1)
    
    return 0 if (num_tp_fp == 0) else num_tp / num_tp_fp

def flat_recall(preds, labels):
    pred_flat = np.argmax(preds, axis=2).flatten()
    labels_flat = labels.flatten()
    return np.sum((pred_flat == labels_flat) & (pred_flat == 1)) / np.sum(labels_flat == 1)

def flat_f1(precision, recall):
    mole = 2 * precision * recall
    deno = precision + recall
    return 0 if (deno == 0) else (mole / deno)

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
10. 开始微调
"""
# Number of epochs (passes through the dataset) to train the model for.
num_epochs = 15
max_grad_norm = 1.0

val_f1s = []
val_ps = []
val_rs = []
val_accs = []

for epoch in range(num_epochs):
    print('='*100)
    print("Starting epoch {}".format(epoch + 1))
    model.train()
    tr_loss = 0
    nb_tr_steps = 0

    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        # 前向过程
        loss = model(b_input_ids, token_type_ids=None,
                     attention_mask=b_input_mask, labels=b_labels)[0]
        # print(loss.item())
	    # 后向过程
        loss.backward()
        # 损失
        tr_loss += float(loss.item())
        nb_tr_steps += 1
        # 减小梯度 https://www.cnblogs.com/lindaxin/p/7998196.html
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
        # 更新参数
        optimizer.step()
        model.zero_grad()
    
    # 保存模型
    # print('saving model for epoch {}'.format(epoch + 1))
    # if not os.path.exists(output_dir + 'model_epoch{}'.format(epoch + 1)):
    #     os.mkdir(output_dir + 'model_epoch{}'.format(epoch + 1))
    # model_to_save = model.module if hasattr(model, 'module') else model
    # model_to_save.save_pretrained(output_dir + 'model_epoch{}'.format(epoch + 1))
    # print('epoch {} finished'.format(epoch + 1))

    #打印每个epoch的损失
    print("Train loss: {}".format(tr_loss/nb_tr_steps))
    
    # 验证过程
    model.eval()
    eval_loss, eval_accuracy, eval_precision, eval_recall, eval_f1 = 0, 0, 0, 0, 0
    nb_eval_steps = 0
    predictions , true_labels = [], []
    for batch in valid_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            tmp_eval_loss = model(b_input_ids, token_type_ids=None,
                                  attention_mask=b_input_mask, labels=b_labels)
            logits = model(b_input_ids, token_type_ids=None,
                           attention_mask=b_input_mask)

        tmp_eval_loss = tmp_eval_loss[0]
        logits = logits[0]

        logits = logits.detach().cpu().numpy()#detach的方法，将variable参数从网络中隔离开，不参与参数更新
        label_ids = b_labels.cpu().numpy()

        # print("np.argmax(logits, axis=2)", np.argmax(logits, axis=2))

        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
        true_labels.append(label_ids)
        # 计算accuracy 和 loss
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        tmp_eval_precision = flat_precision(logits, label_ids)
        tmp_eval_recall = flat_recall(logits, label_ids)
        tmp_eval_f1 = flat_f1(tmp_eval_precision, tmp_eval_recall)

        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy
        eval_precision += tmp_eval_precision
        eval_recall += tmp_eval_recall
        eval_f1 += tmp_eval_f1
        nb_eval_steps += 1
    
    val_accs.append(eval_accuracy/nb_eval_steps)
    val_ps.append(eval_precision/nb_eval_steps)
    val_rs.append(eval_recall/nb_eval_steps)
    val_f1s.append(eval_f1/nb_eval_steps)

    # 打印信息
    print("Validation loss: {}".format(eval_loss/nb_eval_steps))
    print("Validation Accuracy: {}".format(val_accs[epoch]))
    print("Validation Precision: {}".format(val_ps[epoch]))
    print("Validation Recall: {}".format(val_rs[epoch]))
    print("F1-Score: {}".format(val_f1s[epoch]))

# 保存最终模型
print('='*100)
print('training finished')
print("aver_f1: {}".format(sum(val_f1s) / num_epochs))
print("aver_precision: {}".format(sum(val_ps) / num_epochs))
print("aver_recall: {}".format(sum(val_rs) / num_epochs))
print("aver_accuracy: {}".format(sum(val_accs) / num_epochs))

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
model_to_save = model.module if hasattr(model, 'module') else model
model_to_save.save_pretrained(output_dir)
