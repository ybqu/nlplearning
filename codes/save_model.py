import os
import torch
from transformers import BertTokenizer, BertForMaskedLM

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

model_dir = './bert-base-uncased/'
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

tokenizer.save_pretrained(model_dir)

# 打印模型所有参数
dict_b = torch.load(os.path.join(model_dir, 'pytorch_model.bin'))
for key in dict_b.keys():
    print(key)