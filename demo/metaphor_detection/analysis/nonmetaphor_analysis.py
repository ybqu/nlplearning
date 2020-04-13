#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   data_analysis.py
@Time    :   2020/03/30 19:22:21
@Author  :   Quxiansen 
@Version :   1.0
@Contact :   2191002033@cnu.edu.cn
@License :   
@Desc    :   对mask-lm数据进行分析
'''

# here put the import lib
import math
import csv

def main():

    with open('./EnWords.csv', encoding='latin-1') as f:
        lines = csv.reader(f)
        next(lines)
        words_list = {line[0]:i for i, line in enumerate(lines)}

    target_words_list = []
    with open('../mlm/non-metaphor/vua.out', 'r+') as f:
        lines = f.readlines()
        for line in lines:
            raw_words_list = [words.split(',') for words in line.split()[2:]]
            temp_words_list = []
            for words in raw_words_list:
                temp = [word for word in words if word in words_list]
                temp_words_list.append(','.join(temp))

            temp = [line.split()[0], line.split()[1], ' '.join(temp_words_list)]
            target_words_list.append(temp)

    with open('./vua.out', 'w+') as f:
        temp = [' '.join(words) for words in target_words_list]
        f.write('\n'.join(temp))

    print('保存完成！')


if __name__ == "__main__":
    main()