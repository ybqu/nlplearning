## Transformer-XL介绍

[【论文地址】Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860)

[TOC]

---

### 1. Vanilla Transformer

Transformer需要对输入序列设置一个**固定长度**，比如在BERT中，默认长度是512。

+ 如果文本序列长度短于固定长度，可以通过填充的方式来解决;

+ 如果序列长度超过固定长度，处理起来就比较麻烦。

一种处理方式，就是将文本划分为多个segments。训练的时候，对每个segment单独处理，segments之间没有联系，如下图(a)所示。这存在两个问题：

+ 因为segments之间独立训练，所以不同的token之间，最长的依赖关系，就取决于segment的长度；
+ 出于效率的考虑，在划分segments的时候，不考虑句子的自然边界，而是根据固定的长度来划分序列，导致分割出来的segments在语义上是不完整的。

在预测的时候，会对固定长度的segment做计算，一般取最后一个位置的隐向量作为输出。为了充分利用上下文关系，在每做完一次预测之后，就对整个序列向右移动一个位置，再做一次计算，如上图(b)所示，这导致计算效率非常低。

<img src="https://ybqu.oss-cn-beijing.aliyuncs.com/img/image-20200829170238375.png" style="width: 600px" title=""/>

---

### 2. Segment-Level Recurrence

为了解决上面提到的问题，在Transfromer的基础上，Transformer-XL提出了一个改进，在对当前segment进行处理的时候，**缓存并利用**上一个segment中所有layer的隐向量序列，而且上一个segment的所有隐向量序列只参与前向计算，不再进行反向传播，这就是所谓的segment-level Recurrence。

#### 2.1 计算方法

将两个连续的segments表示为 $s_\tau=[x_{\tau,1}, x_{\tau,2},\cdots, x_{\tau,L}]$，$s_{\tau+1}=[x_{\tau+1,1}, x_{\tau+1,2}, \cdots, x_{\tau+1,L}]$，$L$是序列长度。假设整个模型中，包含N层Transformer，那么每个segment中就有**N组**长度为L的隐向量序列，将第$\tau$个segment的第n层隐向量序列表示为$h^n_\tau\in R^{L\times d}$，$d$是隐向量的维度。那么第$\tau+1$个segment的第n层隐向量序列，可以由下面的一组公式计算得出。

$\widetilde{\mathbf{h}}_{\tau+1}^{n-1}=\left[\mathrm{SG}\left(\mathbf{h}_{\tau}^{n-1}\right) \circ \mathbf{h}_{\tau+1}^{n-1}\right]$

$\mathbf{q}_{\tau+1}^{n}, \mathbf{k}_{\tau+1}^{n}, \mathbf{v}_{\tau+1}^{n}=\mathbf{h}_{\tau+1}^{n-1} \mathbf{W}_{q}^{\top}, \widetilde{\mathbf{h}}_{\tau+1}^{n-1} \mathbf{W}_{k}^{\top}, \widetilde{\mathbf{h}}_{\tau+1}^{n-1} \mathbf{W}_{v}^{\top}$

$\mathbf{h}_{\tau+1}^{n}= Transformer-Layer \left(\mathbf{q}_{\tau+1}^{n}, \mathbf{k}_{\tau+1}^{n}, \mathbf{v}_{\tau+1}^{n}\right).$

+ SG是stop-gradient，不再对 $s_\tau$的隐向量做反向传播。$\tilde{h}^{n-1}_{\tau+1}$是对两个隐向量序列沿长度方向的拼接，[]内两个隐向量的维度都是$L\times d$，拼接之后的向量维度是$2L\times d$。3个W分别对应query，key和value的转化矩阵。注意q的计算方式不变，只使用当前segment中的隐向量，计算得到的q序列长度仍然是$L$。k和v采用拼接之后的$\tilde{h}$来计算，计算出来的序列长度是2L。之后的计算就是标准的Transformer计算。计算出来的第n层隐向量序列长度仍然是L，而不是2L。Transformer的输出隐向量序列长度取决于query的序列长度，而不是key和value。

训练和预测过程如下图所示。这张图上有一个点需要注意，在当前segment中，第n层的每个隐向量的计算，都是利用下一层中包括当前位置在内的，连续前L个长度的隐向量，这是在上面的公式组中没有体现出来的，也是文中没有明说的。每一个位置的隐向量，除了自己的位置，都跟下一层中前$L-1$个位置的token存在依赖关系，而且每往下走一层，依赖关系长度会增加$L-1$，如下图中Evaluation phase所示，所以最长的依赖关系长度是$N(L-1)$，N是模型中layer的数量。N通常要比L小很多，比如在BERT中，N=12或者24，L=512，依赖关系长度可以近似为$O(N\times L)$ 。在对长文本进行计算的时候，可以缓存上一个segment的隐向量的结果，不必重复计算，大幅提高计算效率。

![](https://ybqu.oss-cn-beijing.aliyuncs.com/img/image-20200829174600870.png)

+ 上文中，我们只保存了上一个segment，实际操作的时候，可以保存尽可能多的segments，只要内存或者显存放得下。论文中的试验在训练的时候，只缓存一个segment，在预测的时候，会缓存多个segments。

---

### 3. Relative Position Encodings

在vanilla Transformer中，为了表示序列中token的顺序关系，在模型的输入端，对每个token的输入embedding，加一个位置embedding。位置编码embedding或者采用正弦\余弦函数来生成，或者通过学习得到。在Transformer-XL中，这种方法行不通，每个segment都添加相同的位置编码，多个segments之间无法区分位置关系。Trm-XL**放弃使用绝对位置编码，而是采用相对位置编码**，在计算当前位置隐向量的时候，考虑与之依赖token的相对位置关系。具体操作是，在算attention score的时候，只考虑query向量与key向量的相对位置关系，并且将这种相对位置关系，加入到每一层Transformer的attention的计算中。

+ 绝对位置编码：
  $$
  \mathbf{A}_{i, j}^{\mathrm{abs}}=\underbrace{\mathbf{E}_{x_{i}}^{\top} \mathbf{W}_{q}^{\top} \mathbf{W}_{k} \mathbf{E}_{x_{j}}}_{(a)}+\underbrace{\mathbf{E}_{x_{i}}^{\top} \mathbf{W}_{q}^{\top} \mathbf{W}_{k} \mathbf{U}_{j}}_{(b)}
  \\
  +\underbrace{\mathbf{U}_{i}^{\top} \mathbf{W}_{q}^{\top} \mathbf{W}_{k} \mathbf{E}_{x_{j}}}_{(c)}+\underbrace{\mathbf{U}_{i}^{\top} \mathbf{W}_{q}^{\top} \mathbf{W}_{k} \mathbf{U}_{j}}_{(d)}
  $$

  + $E_x$表示token的输入embedding，$U$是绝对位置编码embedding，两个W分别是query矩阵和key矩阵。公式(1)是对$(E_{x_i}+U_i)W_qW_k(E_{x_j}+U_j)$做分解。

+ 相对位置编码：

$$
\mathbf{A}_{i, j}^{\mathrm{rel}}=\underbrace{\mathbf{E}_{x_{i}}^{\top} \mathbf{W}_{q}^{\top} \mathbf{W}_{k, E} \mathbf{E}_{x_{j}}}_{(a)}+\underbrace{\mathbf{E}_{x_{i}}^{\top} \mathbf{W}_{q}^{\top} \mathbf{W}_{k, R} \mathbf{R}_{i-j}}_{(b)}
\\+\underbrace{u^{\top} \mathbf{W}_{k, E} \mathbf{E}_{x_{j}}}_{(c)}+\underbrace{v^{\top} \mathbf{W}_{k, R} \mathbf{R}_{i-j}}_{(d)}
$$

​		将绝对位置编码$U$，替换成了相对位置编码$R_{i-j}$。 因为i只利用之前的序列，所以i-j>=0。		其次，对于所依赖的key向量序列，query向量$U_iW_q$都是固定的，因此将公式组(1)(c)中		的$U_iW_q$替换为$u\in R^d$，将上面(d)中的 $U_iW_q$替换为 $v\in R^d$，u和v都通过学习得到。最		后，将 $W_k$矩阵再细分成两组矩阵 $W_{k,R}$和 $W_{k,R}$，分别生成基于内容的key向量和基于位置		的key向量。可以仔细思考一下每一项中的依赖关系。

​		相对位置关系用一个位置编码矩阵 $R\in \mathbf{R}^{L_{max}\times d}$来表示，第i行表示相对位置间隔为i的位		置向量。论文中强调$R$采用正弦函数生成，而不是通过学习得到的，好处是预测时，可以		使用比训练距离更长的位置向量。

Transformer-XL的完整计算公式，如下所示，只有前3行与vanilla Transformer不同，后3行是一样的。第3行公式中，计算A的时候直接采用query向量，而不再使用 $E_xW_q$ 表示。最后需要注意的是，每一层在计算attention的时候，都要包含相对位置编码。而在vanilla Transformer中，只有在输入embedding中才包含绝对位置编码，在中间层计算的时候，是不包含位置编码的。
$$
\widetilde{\mathbf{h}}_{\tau}^{n-1}=\left[\mathrm{SG}\left(\mathbf{m}_{\tau}^{n-1}\right) \circ \mathbf{h}_{\tau}^{n-1}\right]
\\\mathbf{q}_{\tau}^{n}, \mathbf{k}_{\tau}^{n}, \mathbf{v}_{\tau}^{n}=\mathbf{h}_{\tau}^{n-1} \mathbf{W}_{q}^{n \top}, \widetilde{\mathbf{h}}_{\tau}^{n-1} \mathbf{W}_{k, E}^{n} \mathbf{,} \widetilde{\mathbf{h}}_{\tau}^{n-1} \mathbf{W}_{v}^{n \top}
\\\mathbf{A}_{\tau, i, j}^{n}=\mathbf{q}_{\tau, i}^{n} \mathbf{T}_{\mathbf{k}, j}+\mathbf{q}_{\tau, i}^{n} \mathbf{T} \mathbf{W}_{k, R}^{n} \mathbf{R}_{i-j}+u^{\top} \mathbf{k}_{\tau, j}+v^{\top} \mathbf{W}_{k, R}^{n} \mathbf{R}_{i-j}
\\ \mathbf{a}_{\tau}^{n}=\text { Masked-Softmax }\left(\mathbf{A}_{\tau}^{n}\right) \mathbf{v}_{\tau}^{n}
\\\mathbf{o}_{\tau}^{n}=\text { LayerNorm }\left(\operatorname{Linear}\left(\mathbf{a}_{\tau}^{n}\right)+\mathbf{h}_{\tau}^{n-1}\right)
\\\mathbf{h}_{\tau}^{n}= \text{Positionwise-Feed-Forward }\left(\mathbf{o}_{\tau}^{n}\right)
$$

---

### 总结

Transformer-XL为了解决长序列的问题，对上一个segment做了缓存，可供当前segment使用，但是也带来了位置关系问题，为了解决位置问题，又打了个补丁，引入了相对位置编码。

---

### 参考

[【1】Transformer-XL介绍](https://zhuanlan.zhihu.com/p/84159401)