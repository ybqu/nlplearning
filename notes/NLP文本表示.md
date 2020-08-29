## NLP 文本表示

[TOC]

---

### 1. One-hot向量

假设英文中常用的单词有3万个，那么就用一个3万维的向量表示这个词，所有位置都置0，当我们想表示apple这个词时，就在对应位置设置1，如下图所示：

<img src="https://ybqu.oss-cn-beijing.aliyuncs.com/img/image-20200829151529344.png" style="width: 300px" title=""/>

+ 存在的问题：
  + 高维稀疏，高维是指有多少个词，就需要多少个维度的向量，稀疏是指，每个向量中大部分值都是0；
  + 向量没有任何含义。

---

### 2. Word Embedding

用一个低维稠密的向量去表示一个词，如下图所示。通常这个向量的维度在几百到上千之间，相比one-hot的维度就低了很多。词与词之间可以通过相似度或者距离来表示关系，相关的词向量相似度比较高，或者距离比较近，不相关的词向量相似度低，或者距离比较远，这样词向量本身就有了含义。

词向量可以通过一些无监督的方法学习得到，比如CBOW或者Skip-Gram等，可以预先在语料库上训练出词向量，以供后续的使用。顺便提一句，在图像中就不存在表示方法的困扰，因为图像本身就是数值矩阵，计算机可以直接处理。

<img src="https://ybqu.oss-cn-beijing.aliyuncs.com/img/image-20200829151658553.png" style="width: 300px" title=""/>

#### 2.1 Elmo

#### 2.2 Glove

#### 2.3 Word2Vec

---

### 参考

[【1】自然语言处理中的Transformer和BERT](https://zhuanlan.zhihu.com/p/53099098)

