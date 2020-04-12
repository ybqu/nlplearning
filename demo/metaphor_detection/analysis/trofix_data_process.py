import csv
import ast
import codecs

def data_write_csv(file_name, datas):#file_name为写入CSV文件的路径，datas为要写入数据列表
    file_csv = codecs.open(file_name,'w+','utf-8')#追加
    writer = csv.writer(file_csv)
    for data in datas:
        writer.writerow(data)
    print("保存成功！")

if __name__ == "__main__":
    
    """ 处理数据 """
    raw_analysis_vua = [] # [[sen_ix, label_seq, pos_seq], ..., ]

    with open('./data/TroFi-X/TroFi-X_formatted_svo.csv', encoding='latin-1') as f:
        lines = csv.reader(f)
        next(lines)
        for line in lines:
            if int(line[5]) == 0:
                continue
            m_sen = line[3].split(' ')
            index = m_sen.index(line[1])
            m_sen[index] = '[MASK]'
            label_seq = [0] * len(m_sen)
            label_seq[index] = 1

            raw_analysis_vua.append([line[3], str(label_seq), ' '.join(m_sen)])
    
    data_write_csv('./trofix_masked.csv', raw_analysis_vua)