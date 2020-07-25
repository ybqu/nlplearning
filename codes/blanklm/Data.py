#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Data.py
@Time    :   2020/06/08 21:03:39
@Author  :   Aiken 
@Version :   1.0
@Contact :   2191002033@cnu.edu.cn
@License :   
@Desc    :   数据处理
'''

# here put the import lib


import collections
import os
import numpy as np
import random
import math

MAX_LENGTH = 100


class Vocab:
    def __init__(self, vocab_file, content_file, vocab_size=50000):
        self._word2id = {}
        self._id2word = []
        self._wordcount = {}
        self._voc_size = 0
        if not os.path.exists(vocab_file):
            self.build_vocab(content_file, vocab_file)
        self.load_vocab(vocab_file, vocab_size)

    # def _read_words(self, corpus_file):
    #     with open(corpus_file, 'r+') as r:
    #         return r.read().replace('\n', '<eos>').split()
    
    def build_vocab(self, corpus_file, vocab_file):
        # words = _read_words(corpus_file)
        with open(corpus_file, 'r+') as r:
            words = r.read().replace('\n', '<eos>').split()
        counter = collections.Counter(words)
        cnt_pairs = sorted(counter.items(), key=lambda  x: (-x[1], x[0]))

        with open(vocab_file, 'w') as w:
            for pair in cnt_pairs:
                w.write(pair[0] + '\t' + str(pair[1]) + '\n')

    def load_vocab(self, vocab_file, vocab_size):
        with open(vocab_file, 'r') as r:
            lines = r.readlines()
            for line in lines:
                term_ = line.strip().split('\t')
                if len(term_) < 2:
                    continue
                word, cnt = term_
                assert word not in self._word2id
                self._word2id[word] = len(self._word2id)
                self._id2word.append(word)
                self._wordcount[word] = int(cnt)
                if len(self._word2id) >= vocab_size:
                    break
            assert len(self._word2id) == len(self._id2word)

    def word2id(self, word):
        return self._word2id[word] if word in self._word2id else '<unk>'

    def id2word(self, word_id):
        return self._id2word[word_id]


class Example:
    def __init__(self, content, target, vocab, is_train):
        self.ori_content = content
        if is_train:
            self.ori_target = target
        else:
            self.ori_targets = target
    
    def bow(self, content, maxlen=MAX_LENGTH):
        bow = {}


class Batch:
    def __init__(self, example_list, is_train, model):
        max_len = MAX_LENGTH
        self.model = model
        self.is_train = is_train
        self.examples = example_list

        if model == 'blm':
            pass

    def get_length(slef, examples, max_len):
        length = []
        for e in examples:
            if len(e) > max_len:
                length.append(max_len)
            else:
                length.append(len(e))
        assert len(length) == len(examples)
        return length

    def to_tensor(self):
        if self.model == 'blm':
            pass

    @staticmethod
    def padding(batch, max_len, limit_length=True):
        if limit_length:
            max_len = min(max_len, MAX_LENGTH)
        result = []
        mask_batch = []
        for s in batch:
            l = copy.deepcopy(s)
            m = [1. for _ in range(len(l))]
            l = l[:max_len]
            m = m[:max_len]
            while len(1) < max_len:
                l.append(0)
                m.append(0.)
            result.append(l)
            mask_batch.append(l)
            mask_batch.append(m)
        return result, mask_batch


# class DataLoader:
#     def __init__(self):
#         pass


def dataprocess(corpus_file, tokenizer, mr=0.2):
    """ 数据处理 """
    sents = []
    labels = []
    postions = []
    with open(corpus_file, 'r') as r:
        lines = r.readlines()
        for line in lines:
            label = []
            sent_list = line.strip().split(' ')
            try:
                mask_pos = random.sample(range(len(sent_list)-1), math.ceil(len(sent_list)*mr))
            except:
                continue
            for i, mask in enumerate(mask_pos):
                sent_list[mask] = '_'
                if mask_pos[i] != 0:
                    l_label = 1 if sent_list[mask - 1] == '_' else 0
                else:
                    l_label = 0
                if mask_pos[i] != len(sent_list) - 1:
                    r_label = 1 if sent_list[mask + 1] == '_' else 0
                else:
                    r_label = 0
                label.append((l_label, r_label))
            
            labels.append(label[::-1])
            postions.append(mask_pos[::-1])

                    
            # sent = [token if i not in mask_pos else '_' for i, token in enumerate(sent_list)]
            sent = [v for i, v in enumerate(sent_list) if i == 0 or v != sent_list[i-1]]

            sents.append(' '.join(sent))


def main():
    dataprocess('./data/PennTreeBank/ptb.test.txt')


if __name__ == "__main__":
    main()
