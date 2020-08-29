## XLNet模型

[TOC]

---

### 1. 模型简介

xlnet作为bert的升级模型，主要在以下三个方面进行了优化：

- 采用AR模型替代AE模型，解决mask带来的负面影响；
- 双流注意力机制；
- 引入transformer-xl；

---

### 2. AR与AE语言模型（nlp预训练模型）

+ **AR(autoregressive)语言模型：** 
  + AR模型的主要任务在于**评估语料的概率分布**，例如，给定一个序列$X=(x_1, \ldots, x_T)$，AR模型就是在计算其极大似然估计$p(X)=\prod^T_{t=1}p(x_t|x_{<t})$，即已知$x_t$之前的序列，预测 $x_t$ 的值，当然也可以反着来 $p(X)=\prod^T_{t=1}p(x_t|x_{>t})$ ，即已知$x_t$之后的序列，预测 $x_t$ 的值；
  + **缺点：**该模型是单向的，而我们更希望的是根据上下文来预测目标，而不单是上文或者下文，之前open AI提出的GPT就是采用的AR模式，包括GPT2.0也是该模式。
+ **AE(autoencoding)语言模型：**
  + AE模型采用的就是**以上下文的方式**，最典型的成功案例就是BERT。例如：BERT的预训练包括了两个任务，Masked Language Model与Next Sentence Prediction，Next Sentence Prediction即判断两个序列的推断关系，Masked Language Model采用了一个标志位[MASK]来随机替换一些词，再用[MASK]的上下文来预测[MASK]的真实值，BERT的最大问题也是处在这个MASK的点，因为在微调阶段，没有MASK这就导致预训练和微调数据的不统一，从而引入了一些人为误差。
+ **排列语言模型**
  + 文中提出的**排列语言模型**，该模型不再对传统的AR模型的序列的值按顺序进行建模，而是最大化所有可能的序列的因式分解顺序的期望对数似然，举例：
    + 假如有一个序列[1,2,3,4]，如果我们的预测目标是3，对于传统的AR模型来说，结果是 $p(3)=\prod^3_{t=1}p(3|x_{<t})$ ，如果采用本文的方法，先对该序列进行因式分解，最终会有24种排列方式，下图是其中可能的四种情况，对于第一种情况因为3的左边没有其他的值，所以该情况无需做对应的计算，第二种情况3的左边还包括了2与4，所以得到的结果是 $p(3)=p(3|2)*p(3|2,4)$ ，后续的情况类似，这样处理过后不但保留了序列的上下文信息，也避免了采用mask标记位，巧妙的改进了bert与传统AR模型的缺点。

<img src="https://ybqu.oss-cn-beijing.aliyuncs.com/img/image-20200829205155132.png" style="width: 600px" title=""/>

---

### 3. 基于目标感知表征的双流自注意力

虽然排列语言模型能满足目前的目标，但是对于普通的transformer结构来说是存在一定的问题的，假设我们要求这样的一个对数似然 $p_\theta(X_{z_t}|x_{z_{<t}})$，如果采用标准的softmax的话，那么
$$
p_\theta(X_{z_t}|x_{z_{<t}})=\frac{exp(e(x)^Th_\theta(x_{z_{<t}}))}{\sum_{x^\prime}exp(e(x^\prime)^Th_\theta(x_{z_{<t}}))}
$$
其中 $h_\theta(x_{z_{<t}})$ 表示的是添加了mask后的transformer的输出值，可以发现 $h_\theta(x_{z_{<t}})$ 并不依赖于其要预测的内容的位置信息，**因为无论预测目标的位置在哪里，因式分解后得到的所有情况都是一样的，并且transformer的权重对于不同的情况是一样的，因此无论目标位置怎么变都能得到相同的分布结果**，如下图所示，假如我们的序列index表示为[1,2,3]，对于目标2与3来说，其因式分解后的结果是一样的，那么经过transformer之后得到的结果肯定也是一样的。

<img src="https://ybqu.oss-cn-beijing.aliyuncs.com/img/image-20200829210559655.png" style="width: 100px" title=""/>

这就导致模型没法得到正确的表述，为了解决这个问题，论文中提出来新的分布计算方法，来实现目标位置感知
$$
p_{\theta}\left(X_{z_{t}}=x \mid x_{z_{<t}}\right)=\frac{\exp \left(e(x)^{T} g_{\theta}\left(x_{z_{<t}}, z_{t}\right)\right)}{\sum_{x^{\prime}} \exp \left(e\left(x^{\prime}\right)^{T} g_{\theta}\left(x_{z_{<t}}, z_{t}\right)\right)}
$$
其中 $g_\theta(x_{z_{<t}}, z_t)$ 是新的表示形式，并且把位置信息 $z_t$ 作为了其输入。

接下来我们就详细来讲解下这个新的表示形式，论文把该方法称为Two-Stream Self-Attention，双流自注意力，该机制需要解决了两个问题：

- 如果目标是预测 $x_{z_t}$ ，$g_\theta(x_{z_{<t}}, z_t)$ 那么只能有其位置信息$z_t$而不能包含内容信息$x_z$；
- 如果目标是预测其他tokens即 $x_{z_j}，j>t$，那么应该包含$x_{z_t}$的内容信息这样才有完整的上下文信息；

很显然传统的transformer并不满足这样的需求，因此作者采用了两种表述来代替原来的表述，这也是为什么称为**双流**的原因，我们看下这两种不同的表述：

+ `content representation`内容表述，即$h_{\theta}\left(x_{z_{\leq t}}\right)$，下文用$h_{z_t}$表示，该表述和传统的transformer一样，同时编码了上下文和$x_{z_t}$自身

$$
h_{z_{t}}^{(m)}=A t t e n t i o n\left(Q=h_{z_{t}}^{(m-1)}, K V=h_{z \leq t}^{(m-1)} ; \theta\right)
$$

<img src="https://ybqu.oss-cn-beijing.aliyuncs.com/img/image-20200829211529625.png" style="width: 300px" title=""/>

- `query representation`查询表述，即$g_{\theta}\left(x_{z_{<t}}, x_{t}\right)$，下文用$g_{z_t}$表示，该表述包含上下文的内容信息$x_{z_{<t}}$和目标的位置信息$z_t$，但是不包括目标的内容信息$x_{z_t}$，从图中可以看到，K与V的计算并没有包括Q，自然也就无法获取到目标的内容信息，但是目标的位置信息在计算Q的时候保留了下来

$$
g_{z_{t}}^{(m)}=A t t e n t i o n\left(Q=g_{z_{t}}^{(m-1)}, K V=h_{z<t}^{(m-1)} ; \theta\right)
$$

<img src="https://ybqu.oss-cn-beijing.aliyuncs.com/img/image-20200829211911068.png" style="width: 300px" title=""/>

+ 总的计算过程：
  + 首先，第一层的查询流是随机初始化了一个向量即 $g^{(0)}_i=\omega$ ，内容流是采用的词向量即 $h^{(0)}_i=e(x_i)$ ，self-attention的计算过程中两个流的网络权重是共享的，最后在微调阶段，只需要简单的把query stream移除，只采用content stream即可。

<img src="https://ybqu.oss-cn-beijing.aliyuncs.com/img/image-20200829212230019.png" style="width: 400px" title=""/>

---

### 4. 集成Transformer-XL

将transformer-xl的两个最重要的技术点应用了进来，即**相对位置编码**与**片段循环机制**。

#### 4.1 片段循环机制

- transformer-xl的提出主要是为了解决超长序列的依赖问题，对于普通的transformer由于有一个最长序列的超参数控制其长度，对于特别长的序列就会导致丢失一些信息，transformer-xl就能解决这个问题；

- 假设我们有一个长度为1000的序列，如果我们设置transformer的最大序列长度是100，那么这个1000长度的序列需要计算十次，并且每一次的计算都没法考虑到每一个段之间的关系，如果采用transformer-xl，首先取第一个段进行计算，然后把得到的结果的隐藏层的值进行缓存，第二个段计算的过程中，把缓存的值拼接起来再进行计算。该机制不但能保留长依赖关系还能加快训练，因为每一个前置片段都保留了下来，不需要再重新计算，在transformer-xl的论文中，经过试验其速度比transformer快了1800倍；
- 在xlnet中引入片段循环机制其实也很简单，只需要在计算KV的时候做简单的修改，其中 $\tilde{h}^{(m-1)}$ 表示的是缓存值。

$$
h_{z_{t}}^{(m)}=A t t e n t i o n\left(Q=h_{z_{t}}^{(m-1)}, K V=\left[\tilde{h}^{(m-1)}, h_{z \leq t}^{(m-1)}\right] ; \theta\right)
$$

#### 4.2 相对位置编码

+ BERT的position embedding采用的是绝对位置编码，但是绝对位置编码在transformer-xl中有一个致命的问题，因为没法区分到底是哪一个片段里的，这就导致了一些位置信息的损失，这里被替换为了transformer-xl中的相对位置编码。
+ 假设给定一对位置 $i$ 与 $j$ ，如果 $i$ 和 $j$ 是同一个片段里的那么我们令这个片段编码$s_{ij}=s_+$，如果不在一个片段里则令这个片段编码为$s_{ij}=s_-$，这个值是在训练的过程中得到的，也是用来计算attention weight时候用到的，在传统的transformer中 $\text{attention weight}=softmax(\frac{Q\cdot K}{d}V)$，在引入相对位置编码后，首先要计算出$a_{ij}=(q_i+b)^Ts_{s_j}$，其中$b$也是一个需要训练得到的偏执量，最后把得到的$a_{ij}$与传统的transformer的weight相加从而得到最终的attention weight。

#### 4.3 预训练

预训练阶段和BERT差不多，不过去除了Next Sentence Prediction，作者发现该任务对结果的提升并没有太大的影响。输入的值还是 [A, SEP, B, SEP, CLS]的模式，A与B代表的是两个不同的片段。

---

### 参考

[【1】最通俗易懂的XLNET详解](https://blog.csdn.net/u012526436/article/details/93196139)