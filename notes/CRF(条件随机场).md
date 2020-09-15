## CRF(条件随机场) vs HMM(隐马尔科夫模型)

[TOC]

---

### 1. CRF / HMM 简介

#### 1.1 What is conditional random field?

+ **Random** - 指的是随机变量 $X, Y$ ；
+ **Conditional** - 指的是条件概率 Conditional probability，所以CRF是一个判别式模型。
  + 判别式模型 (discriminative model) ：计算条件概率；
  + 生成式模型 (generative model)： 计算联合概率分布。

#### 1.2 CRF可以做什么?

CRF 是一个序列化标注算法（sequence labeling algorithm），接收一个输入序列如 $X=(x_1, x_2, \ldots, x_n)$ 并且输出目标序列 $Y=(y_1, y_2, \ldots, y_n)$，也能被看作是一种seq2seq模型。这里使用大写 $X, Y$ 表示序列。

+ **词性标注任务：**输入序列为一串单词，输出序列就是相应的词性；

<img src="https://ybqu.oss-cn-beijing.aliyuncs.com/img/image-20200830131047614.png" style="width: 400px" title=""/>

+ **chuking，命名实体识别等**

一般地，输入序列 $X$ 被称为 **observations**, $Y$ 叫作 **states。**于是我们可以将简单版linear CRF的图模型表达出来。

<img src="https://ybqu.oss-cn-beijing.aliyuncs.com/img/image-20200830131407541.png" style="width: 400px" title=""/>

#### 1.3 HMM (Hidden Markov Model)

+ **Generative model**

给定一个观测序列observations, HMM models joint probability $P_r(X, Y)$ 。以词性标注任务举例：
$$
\operatorname{Pr}(T \mid W)=\frac{\operatorname{Pr}(W \mid T) * \operatorname{Pr}(T)}{\operatorname{Pr}(W)}
$$

$$
\begin{aligned}
\operatorname{Pr}(W, T) &=\operatorname{Pr}(W \mid T) * \operatorname{Pr}(T) \\
&=\prod_{i=1}^{n} \operatorname{Pr}\left(w_{i} \mid T\right) * \operatorname{Pr}(T) \\
&=\prod_{i=1}^{n} \operatorname{Pr}\left(w_{i} \mid t_{i}\right) * \operatorname{Pr}(T) \\
&=\prod_{i=1}^{n} \operatorname{Pr}\left(w_{i} \mid t_{i}\right) * \operatorname{Pr}\left(t_{i} \mid t_{i-1}\right)
\end{aligned}
$$

+ $W$and $T$ 指输入单词序列，和输出词性序列。

公式首先应用了贝叶斯公式展开，随后 HMM 做了3个假设进行公式化简：

+ 由词之间 conditionally independent 得到 $\prod^n_{i=1}P_r(\omega_u|T)$ ；
+ 由probability of a word is only dependent on its own tag 得到发射概率 (emission probability) $\prod^n_{i=1}p_r(\omega_i|t_i)$and Markov assumption；
+ 使用bi-gram近似得到转移概率 (Transition probability) $P_r(t_i|t_i-1)$。

这些假设使得HMM能够计算出给定一个词和它可能的词性的联合概率分布。换句话说，HMM假设了两类特征：

+ 当前词词性与上一个词词性的关系；
+ 当前词语和当前词性的关系。

分别对应着转移概率和发射概率。HMM的学习过程就是在训练集中学习这两个概率矩阵(大小为 $(t*t), (\omega*t)$。

稍不同于linear CRF，HMM可以用有向图模型来表示，因为states与observations之间存在着明显的依赖关系。

<img src="https://ybqu.oss-cn-beijing.aliyuncs.com/img/image-20200830133813191.png" style="width: 250px" title=""/>

### 2. Conditional Random Field(CRF)

#### 2.1 From HMM to CRF

---

### 参考

[【1】一文理解条件随机场CRF](https://zhuanlan.zhihu.com/p/70067113)

```

```

