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

    with open('./data/VUAsequence/VUA_seq_formatted_test.csv', encoding='latin-1') as f:
        lines = csv.reader(f)
        next(lines)
        for line in lines:
            label_seq = ast.literal_eval(line[3])
            if 1 not in label_seq:
                continue
            m_lab = [i for i, label in enumerate(label_seq) if label == 1]
            m_sen = ['[MASK]' if i in m_lab else sen for i, sen in enumerate(line[2].split(' '))]
            raw_analysis_vua.append([line[2], str(label_seq), ' '.join(m_sen)])
    
    data_write_csv('./vua_masked.csv', raw_analysis_vua)