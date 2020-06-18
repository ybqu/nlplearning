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


from torch.utils.data import Dataset, DataLoader
import collections
import os
import numpy as np


class Vocab:
    def __init__(self, vocab_file, content_file, vocab_size=5000):
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
    def __init__(self):
        pass


class Batch:
    def __init__(self):
        pass


class DataLoader:
    def __init__(self):
        pass
