## 隐喻任务

+ 数据集：VUA、TroFi

### 相关信息

+ [CodaLab：The Second Shared Task on Metaphor Detection](https://competitions.codalab.org/competitions/22188#learn_the_details)
+ [论文：A Report on the 2018 VUA Metaphor Detection Shared Task](https://www.aclweb.org/anthology/W18-0907/)

### 文档结构
+ analysis
  - ***_data_process.py** -- 对数据集中隐喻词进行[MASK]标记处理，用于 mlm 预测分析；
  - **analysis.py** -- 对 mlm 预测结果进行统计分析；

+ data
  - 数据集目录；

+ mlm
  - **metaphor** -- 隐喻词进行预测，输出隐喻词序列位置；
  - **non-metaphor** -- 隐喻词进行预测，随机输出预测中词的位置；

+ rnn_bert
  - 使用 BertForTokenClassification 进行模型训练；