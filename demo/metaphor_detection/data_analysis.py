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
    mask_list = []
    with open('./trofix_maskedlm.out', 'r+', encoding='utf-8') as f:
        lines = f.readlines()
        lines = lines[5413: ]
        for line in lines:
            index = line.split(' ')[1].rstrip('\n')
            mask_list.append(int(index))

    mask_list.sort()

    slice_num = math.ceil(mask_list[len(mask_list) - 1] / 1000)
    slice_list = [i*1000 - 1 for i in range(slice_num + 1)]

    indexs = [0]

    for i, mask in enumerate(mask_list):
        if mask > slice_list[0]:
            indexs.append(i)
            del slice_list[0]
    
    indexs.append(len(mask_list) - 1)

    result = [mask_list[indexs[i] : indexs[i+1]] for i in range(len(indexs) - 1)]

    result_len = [len(r) for r in result]

    # print(len(result_len), len(slice_list))
    print('{:13}: {}'.format(-1, result_len[0]))
    for i, length in enumerate(result_len):
        if i == 0:
            continue
        print('{:5} ~ {:5}: {}'.format((i-1)*1000, i*1000 - 1, result_len[i]))
    
    mlist = mask_list[mask_list.count(-1):]
    index_sum = 0
    for mask in mlist:
        index_sum += mask

    print('{}{}'.format('平均排名：', index_sum / len(mlist)))

if __name__ == "__main__":
    main()