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

def main():
    metaphor_indexs = []
    ave_indexs = []
    len_words = []
    with open('../mlm/non-metaphor/trofix.out', 'r+', encoding='utf-8') as f:
        lines = f.readlines()
        for i, line in enumerate(lines[2:]):
            indexs = line.split(' ')[1].rstrip('\n')
            indexs = [int(idx) for idx in indexs.split(',')]

            len_words.append(len(indexs) - 1)
            
            metaphor_indexs.append(int(indexs[0]))
            ave_indexs.append(sum(indexs[1:]) // len_words[i])


    metaphor_indexs.sort()
    ave_indexs.sort()


    print('max_预测词：{} min_预测词：{}'.format(max(len_words), min(len_words)))

    meta_slice_nums = math.ceil(metaphor_indexs[len(metaphor_indexs) - 1] / 1000)
    slice_nums = math.ceil(ave_indexs[len(ave_indexs) - 1] / 1000)
    meta_slice_list = [i*1000 - 1 for i in range(meta_slice_nums + 1)]
    slice_list = [i*1000 - 1 for i in range(slice_nums + 1)]

    mpos_indexs = [0]
    pos_indexs = []

    for i, idx in enumerate(metaphor_indexs):
        if idx > meta_slice_list[0]:
            mpos_indexs.append(i)
            del meta_slice_list[0]
    
    for i, idx in enumerate(ave_indexs):
        if idx > slice_list[0]:
            pos_indexs.append(i)
            del slice_list[0]
    
    mpos_indexs.append(len(metaphor_indexs) - 1)
    pos_indexs.append(len(ave_indexs) - 1)

    mresult_len = [int(mpos_indexs[i+1] - mpos_indexs[i]) for i in range(len(mpos_indexs) - 1)]

    result_len = [int(pos_indexs[i+1] - pos_indexs[i]) for i in range(len(pos_indexs) - 1)]

    mresult_len[len(mresult_len)-1] += 1
    result_len[len(result_len)-1] += 1

    print('隐喻词：')
    print('{:13}: {}'.format(-1, mresult_len[0]))
    for i, length in enumerate(mresult_len):
        if i == 0:
            continue
        print('{:5} ~ {:5}: {}'.format((i-1)*1000, i*1000 - 1, length))

    print('非隐喻词：')
    # print('{:13}: {}'.format(-1, result_len[0]))
    for i, length in enumerate(result_len):
        print('{:5} ~ {:5}: {}'.format((i)*1000, (i + 1)*1000 - 1, length))

    print('非隐喻词共：{} 隐喻词共：{}'.format(sum(len_words), len(metaphor_indexs)))
    
if __name__ == "__main__":
    main()