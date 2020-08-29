## NLP三大特征提取器全梳理：RNN vs CNN vs Transformer

[TOC]

******

### 1. RNN(循环神经网络)

+ RNN和CNN（卷积神经网络）的关键区别在于，它是个序列的神经网络，即前一时刻的输入和后一时刻的输入是有关系的。

#### 1.1 RNN结构

+ 简单的循环神经网络：

<img src="https://ybqu.oss-cn-beijing.aliyuncs.com/img/image-20200829123310185.png" style="width:300px" title="简单的循环神经网络"/>

+ 具有多个输入的循环神经网络：

  <img src="https://ybqu.oss-cn-beijing.aliyuncs.com/img/image-20200829123651074.png" style="width:700px" title="具有多个输入的循环神经网络"/>

+ 将RNN以时间序列展开：

  <img src="https://ybqu.oss-cn-beijing.aliyuncs.com/img/image-20200829124002345.png" style="width:700px" title="RNN时间序列展开"/>

#### 1.2 RNN存在的问题

+ 梯度消失/梯度爆炸：因为RNN是采用线性序列结构进行传播的，这种方式给反向传播优化带来了困难，容易导致梯度消失以及梯度爆炸等问题；
+ 计算效率低：因为 t 时刻的计算依赖 t-1 时刻的隐层计算结果，而 t-1 时刻的结果又依赖于 t-2 时刻的隐层计算结果……，因此用 RNN 进行自然语言处理时，只能逐词进行，无法执行并行运算，从而导致RNN很难具备高效的并行计算能力，工程落地困难。

为了解决上述问题，后来研究人员引入了 LSTM 和 GRU，获得了很好的效果。

---

### 2. CNN（卷积神经网络）

CNN 不仅在计算机视觉领域应用广泛，在 NLP 领域也备受关注。

从数据结构上来看，CNN 输入数据为文本序列，假设句子长度为 $ n $，词向量的维度为 $ d $，那么输入就是一个 $ n \times d $ 的矩阵。显然，该矩阵的行列「像素」之间的相关性是不一样的，矩阵的同一行为一个词的向量表征，而不同行表示不同的词。

#### 2.1 CNN结构

要让卷积网络能够正常地「读」出文本，我们就需要使用一维卷积。Kim 在 2014 年首次将 CNN 用于 NLP 任务中，其网络结构如下图所示：

<img src="https://ybqu.oss-cn-beijing.aliyuncs.com/img/image-20200829125639015.png" style="width:600px" title="CNN结构图"/>

从图中可以看到，卷积核大小会对输出值的长度有所影响。但经过池化之后，可映射到相同的特征长度（如上图中深红色卷积核是 4 × 5，对于输入值为 7 × 5 的输入值，卷积之后的输出值就是 4 × 1，最大池化之后就是 1 × 1；深绿色的卷积核是 3 × 5，卷积之后的输出值是 5 × 1，最大池化之后就是 1 × 1）。之后将池化后的值进行组合，就得到 5 个池化后的特征组合。

+ **优点：**无论输入值的大小是否相同（由于文本的长度不同，输入值不同），要用相同数量的卷积核进行卷积，经过池化后就会获得相同长度的向量（向量的长度和卷积核的数量相等），这样接下来就可以使用全连接层了（全连接层输入值的向量大小必须一致）。

#### 2.2 特征提取过程

+ 完整的卷积神经网络

~~~flow
```flow
st=>start: 输入层
op1=>operation: 卷积层
op2=>operation: 池化层
op3=>operation: 全连接层
sub1=>subroutine: 全连接层
e=>end: 输出层
st(right)->op1(right)->op2(right)->op3(right)->e(right)
```
~~~

+ 卷积层

在卷积层中，**卷积核**具有非常重要的作用，CNN 捕获到的特征基本上都体现在卷积核里。卷积层包含多个卷积核，每个卷积核提取不同的特征。以单个卷积核为例，假设卷积核的大小为 $ d \times k$，其中 $ d $ 是卷积核指定的窗口大小，$ d $ 是 Word Embedding 长度。卷积窗口依次通过每一个输入，它捕获到的是单词的 k-gram 片段信息，这些 k-gram 片段就是 CNN 捕获到的特征，$ k $ 的大小决定了 CNN 能捕获多远距离的特征。

+ 池化层

通常采用最大池化（max-pooling）方法。如下图所示，执行最大池化方法时，窗口的大小是 2×2，使用窗口滑动，在 2×2 的区域上保留数值最大的特征，由此可以使用最大池化将一个 4×4 的特征图转换为一个 2*2 的特征图。这里我们可以看出，池化起到了***降维**的作用。

<img src="https://ybqu.oss-cn-beijing.aliyuncs.com/img/image-20200829132329195.png" style="width:600px" title="最大池化"/>

+ **缺点：**
  1. 网络结构不深。它只有一个卷积层，无法捕获长距离特征，卷积层做到 2 至 3 层，就没法继续下去
  2. 池化方法，文本特征经过卷积层再经过池化层，会损失掉很多位置信息。而位置信息对文本处理来说非常重要，因此需要找到更好的文本特征提取器。

---

###  3. Transformer

+ Transformer 是谷歌大脑 2017 年论文**《Attentnion is all you need》**中提出的 seq2seq 模型；
+ **应用方式**：先预训练语言模型，然后把预训练模型适配给下游任务，以完成各种不同任务，如分类、生成、标记等。
+ **优点**：
  + 改进了 RNN 训练速度慢的致命问题，该算法采用 **self-attention** 机制实现快速并行；
  + 可以加深网络深度，不像 CNN 只能将模型添加到 2 至 3 层，这样它能够获取更多全局信息，进而提升模型准确率。

#### 3.1 Transformer结构

<img src="https://ybqu.oss-cn-beijing.aliyuncs.com/img/image-20200829133310438.png" style="width: 400px" title="Transformer结构"/>

Transformer 由两大部分组成：编码器（Encoder） 和解码器（Decoder），每个模块都包含 6 个 block。所有的编码器在结构上都是相同的，负责把自然语言序列映射成为隐藏层，它们含有自然语言序列的表达式，但没有共享参数。然后解码器把隐藏层再映射为自然语言序列，从而解决各种 NLP 问题。

##### 3.1.1 具体步骤：

1. 获取输入单词的词向量$ X $ ，$ X $ 由词嵌入和位置嵌入相加得到。其中词嵌入可以采用 Word2Vec 或 Transformer 算法预训练得到，也可以使用现有的 Tencent_AILab_ChineseEmbedding。由于 Transformer 模型不具备循环神经网络的迭代操作，所以我们需要向它提供每个词的位置信息，以便识别语言中的顺序关系，因此位置嵌入非常重要。模型的位置信息采用 **sin** 和 **cos** 的线性变换来表达：

$$
PE(pos, 2i)=sin(\frac {pos}{10000^{2i/d_{model}}})
$$

$$
PE(pos, 2i+1)=cos(\frac {pos}{10000^{2i/d_{model}}})
$$

​		其中，$ pos $ 表示单词在句子中的位置，比如句子由 10 个词组成，则 $ pos $ 表示 [0-9] 的任意位置，取值范围是 [0, max sequence]；$ i $ 表示词 		向量的维度，取值范围 [0, embedding dimension]，例如某个词向量是 256 维，则 i 的取值范围是 [0-255]；$ d $ 表示 PE 的维度，也就是词向		量的维度，如上例中的 256 维；$ 2i $ 表示偶数的维度（sin）；$ 2i+1 $ 表示奇数的维度（cos）。

​		以上 sin 和 cos 这组公式，分别对应 embedding dimension 维度一组奇数和偶数的序号的维度，例如，0,1 一组，2,3 一组。分别用上面的 		sin 和 cos 函数做处理，从而产生不同的周期性变化，学到位置之间的依赖关系和自然语言的时序特性。

<img src="https://ybqu.oss-cn-beijing.aliyuncs.com/img/image-20200829140219953.png" style="width:500px" title=""/>

2. 将第一步得到的向量矩阵传入编码器，编码器包含 6 个 block ，输出编码后的信息矩阵 C。每一个编码器输出的 block 维度与输入完全一致；

<img src="https://ybqu.oss-cn-beijing.aliyuncs.com/img/image-20200829140358404.png" style="width: 300px" title="encode"/>

3. 将编码器输出的编码信息矩阵 C 传递到解码器中，解码器会根据当前翻译过的单词 $ 1 \sim i $ 依次翻译下一个单词 $ i+1 $，如下图所示：

<img src="https://ybqu.oss-cn-beijing.aliyuncs.com/img/image-20200829140701611.png" style="width: 500px" title="decoder"/>

#### 3.2 Self-Attention 结构

下图展示了 Self-Attention 的结构。在计算时需要用到 Q(查询), K(键值), V(值)。在实践中，Self-Attention 接收的是输入（单词表示向量 $ x $ 组成的矩阵 $ X $）或者上一个 Encoder block 的输出。而 Q, K, V 正是基于 Self-Attention 的输入进行线性变换得到的。

<img src="https://ybqu.oss-cn-beijing.aliyuncs.com/img/image-20200829141314982.png" style="width:300px" title="self-attention"/>



##### 3.2.1 具体实现（举例）

假如我们要翻译一个词组 Thinking Machines，其中 Thinking 的词向量用 $ x_1 $ 表示，Machines 的词向量用 $ x_2 $ 表示。

1. 定义一个$ W^Q $矩阵（随机初始化，通过训练得到），将embedding和$ W^Q $矩阵做乘法，得到查询向量q，假设输入embedding是512维，在下图中我们用4个小方格表示，输出的查询向量是64维，下图中用3个小方格以示不同。然后类似地，定义$ W^K $和$ W^V $矩阵，将embedding和$ W^K $做矩阵乘法，得到键向量k；将embeding和$ W^V $做矩阵乘法，得到值向量v。对每一个embedding做同样的操作，那么每个输入就得到了3个向量，查询向量，键向量和值向量。需要注意的是，查询向量和键向量要有相同的维度，值向量的维度可以相同，也可以不同，但一般也是相同的。

<img src="https://ybqu.oss-cn-beijing.aliyuncs.com/img/image-20200829142057250.png" style="width: 400px" title=""/>

2. 接下来我们计算每一个embedding的输出，以第一个词Thinking为例，参看下图。用查询向量$q_1$跟键向量$k_1$和$k_2$分别做点积，得到112和96两个数值。这也是为什么前文提到查询向量和键向量的维度必须要一致，否则无法做点积。然后除以常数8，得到14和12两个数值。这个常数8是键向量的维度的开方，键向量和查询向量的维度都是64，开方后是8。做这个尺度上的调整目的是为了易于训练。然后把14和12丢到softmax函数中，得到一组加和为1的系数权重，算出来是大约是0.88和0.12。将0.88和0.12对两个值向量$v_1$和$v_2$做加权求和，就得到了Thinking的输出向量$z_1$。类似的，可以算出Machines的输出$z_2$。如果一句话中包含更多的词，也是相同的计算方法。

<img src="https://ybqu.oss-cn-beijing.aliyuncs.com/img/image-20200829142656131.png" style="width: 300px" title=""/>

3. 通过这样一系列的计算，可以看到，现在每个词的输出向量$z$都包含了其他词的信息，每个词都不再是孤立的了。而且每个位置中，词与词的相关程度，可以通过softmax输出的权重进行分析。如下图所示，这是某一次计算的权重，其中线条颜色的深浅反映了权重的大小，可以看到it中权重最大的两个词是The和animal，表示it跟这两个词关联最大。这就是attention的含义，输出跟哪个词关联比较强，就放比较多的注意力在上面。上面我们把每一步计算都拆开了看，实际计算的时候，可以通过矩阵来计算。

<img src="https://ybqu.oss-cn-beijing.aliyuncs.com/img/image-20200829143238219.png" style="width: 300px" title=""/>

<img src="https://ybqu.oss-cn-beijing.aliyuncs.com/img/image-20200829143327422.png" style="width: 300px" title=""/>

---

### 总结

RNN 在并行计算方面存在严重缺陷，但其线性序列依赖性非常适合解决 NLP 任务，这也是为何 RNN 一引入 NLP 就很快流行起来的原因。但是也正是这一线性序列依赖特性，导致它在并行计算方面要想获得质的飞跃，近乎是不可能完成的任务。而 CNN 网络具有高并行计算能力，但结构不能做深，因而无法捕获长距离特征。而Transformer在并行计算能力和长距离特征捕获能力等方面都表现优异。

---

### 补充：

#### - Transformer encoder结构

<img src="https://ybqu.oss-cn-beijing.aliyuncs.com/img/image-20200829144103981.png" style="width: 200px" title=""/>

上图是Transformer encoder的结构。首先是输入word embedding，这里是直接输入一整句话的所有embedding。假设我们的输入是Thinking Machines，每个词对应一个embedding，就有2个embedding。输入embedding需要加上位置编码（Positional Encoding），然后经过一个**Multi-Head Attention**结构，之后是做一个shortcut的处理，就是把输入和输出按照对应位置加起来（残差网络（ResNet）有利于加速训练）。然后经过一个归一化normalization的操作。接着经过一个两层的全连接网络，最后同样是shortcut和normalization的操作。

需要注意的是，每个小模块的输入和输出向量，维度都是相等的，比如，Multi-Head Attention的输入和输出向量维度是相等的，否则无法进行shortcut的操作；Feed Forward的输入和输出向量维度也是相等的；最终的输出和输入向量维度也是相等的。但是Multi-Head Attention和Feed Forward内部，向量维度会发生变化。

#### - Multi-Head Attention结构

+ Self-Attention： 参照上面部分；
+ Multi-Head：
  + 对于同一组输入embedding，我们可以并行做若干组Attention的操作，例如，我们可以进行8组这样的运算，每一组都有$W^Q$，$W^K$，$W^V$矩阵，并且不同组的矩阵也不相同。这样最终会计算出8组输出，我们把8组的输出连接起来，并且乘以矩阵$W^O$做一次线性变换得到输出，$W^O$也是随机初始化，通过训练得到，计算过程如下图所示。这样的好处，一是多个组可以并行计算，二是不同的组可以捕获不同的子空间的信息。

<img src="https://ybqu.oss-cn-beijing.aliyuncs.com/img/image-20200829145432166.png" style="width: 500px" title=""/>

### - Transformer中的position embedding

##### 编码的需求

1. 需要体现同一单词在不同位置的区别；
2. 需要体现一定的先后次序，并且在一定范围内的编码差异不应该依赖于文本的长度，具有一定的不变性；
3. 需要有值域的范围限制。

##### 演变过程

1. 对于一个长度为 $T$ 的文本，想要对其进行位置编码，最简单的方式就是计数，即使用：$PE=pos=0,1,2,\ldots,T-1$ 作为文本每个字的位置编码；
   
   + **缺点**：序列没有上界，如果$T=500$ 那最后一个字的编码就会比第一个位置的编码大太多，带来的问题就是和字嵌入合并以后难免会出现特征在数值上的倾斜和干扰字嵌入的「风头」，对模型可能有一定的干扰。
   
2. 根据上述的分析，位置编码最好具有一定的值域范围限制，所以进一步的思路：$PE=pos/(T-1)$ ，即使用文本长度对位置编码做归一化，这样位置编码就都落在了$[0, 1]$域的范围内；

   + **缺点**：不同长度的位置编码的步长不同，在较短的文本中相邻的两个字的位置编码的差异会和长文本中的相邻数个字的位置编码的差异一致。举例：

     1. 故宫在北京（$T=5$）

        $pos(北)=4/5=0.8$

        $pos(京)=5/5=1$

        即：$pos(京)-pos(北)=0.2$

     2. 故宫再北京市东城区景山。。。（$T=500$）

        $pos(北)=4/500=0.008$

        $pos(京)=5/500=0.01$

        即：$pos(京)-pos(北)=0.002$

        再看一下和北字差0.2的position的位置在哪：

        $pos(?)-pos(北)=0.2\Rightarrow pos(?)=x/500=0.208 \Rightarrow x=104$

        同样的一个词 **北京** ，第二句与北的位置编码的差距为0.2的位置在$T=104$的地方。

        这显然是不合适的，我们关注的位置信息，最核心的其实就是相对的位置关系，如果使用这种方式，那么长文本的相对次序关系就会被稀释。

3. 有界周期函数：在前面的两种做法里面，我们为了体现某个字在句子中的**绝对位置**，使用了一个单调的函数，使得任意后续的字符的位置编码都大于前面的字，如果我们放弃对绝对位置的追求，转而要求位置编码仅仅关注一定范围内的相对次序关系，那么使用一个$sin/cos$函数是很好的选择，因为 $sin/cos$ 函数的周期变化规律非常稳定，所以编码具有一定的不变性。简单的构造可以使用下面的形式。

   $$
   PE(pos)=sin(\frac{pos}{\alpha})
   $$
   其中$\alpha$用来调节位置编码函数的波长，当$\alpha$比较大时，波长比较长，相邻字的位置编码之间的差异比较小。
   但是这样的做法还是有些简陋，周期函数的引入是为了复用位置编码函数的值域，但是这种 $Z \rightarrow [-1, 1]$ 的映射还是太单调：如果$\alpha$比较大，相邻字符之间的位置差异体现得不明显；如果$\alpha$比较小，在长文本中还是可能会有一些不同位置的字符的编码一样，这是因为$[-1, 1]$空间的表现范围有限。既然字嵌入的维度是$d_{model}$，自然也可以使用一个$d_{model}$维向量来表示某个位置编码——$[-1, 1]^{d_{model}}$的表示范围要远大于$[-1, 1]$。

   <u>显然，在**不同维度上应该用不同的函数操纵位置编码**</u>，这样高维的表示空间才有意义。可以为位置编码的每一维赋予不同的$\alpha$；甚至在一些维度将$sin(\cdot)$替换为$cos(\cdot)$...一种构造方法就是论文中的方法了。
   $$
   PE(pos, 2i)=sin(\frac{pos}{10000^{2i/d_{model}}})
   $$

   $$
   PE(pos, 2i+1)=cos(\frac{pos}{10000^{2i/d_{model}}})
   $$

   这里不同维度上$sin/cos$的波长从$2\pi$到$10000\cdot 2\pi$都有，区分了奇偶数维度的函数形式。这使得每一维度上都包含了一定的位置信息，而各个位置字符的位置编码又各不相同。

##### 存在的问题

[【TENER: Adapting Transformer Encoder for Named Entity Recognition】](https://arxiv.org/abs/1911.04474)

+ **property1:**

  对于固定偏移量$k$和位置$t$，$PE^T_tPE_{t+k}$仅取决于$k$，也就是说两个位置编码的点积可以反映两个字间的距离。

  **证明：**

  令$c_i=1/10000^{2i/d_{model}}$，则第 $t$ 个位置的position embedding就是：
  $$
  PE_t=\begin{bmatrix}sin(c_0t) \\ cos(c_0t) \\ \ddots \\ sin(c_{\frac{d}{2}-1}t) \\ cos(c_{\frac{d}{2}-1}t) \end{bmatrix}
  $$
  则：

  $PE^T_tPE_{t+k}=\sum^{\frac{d}{2}-1}_{j=0}[sin(c_jt)sin(c_j(t+k)) + cos(c_jt)cos(c_j(t+k))]$

  $=\sum^{\frac{d}{2}-1}_{j=0}cos(c_j(t-(t+k)))$

  $=\sum^{\frac{d}{2}-1}_{j=0}cos(c_jk)$

  用到的就是一个简单的三角变换公式：$sin(x)sin(y)+cos(s)cos(y)=cos(x-y)$

+ **property2:**

  对于偏移量 $k$和位置 $t$，$PE_{t}^{T}PE_{t-k}=PE_{t}^{T}PE_{t+k}$，意味着这种位置向量的编码是没有方向性。

  **证明：**

  令 $j=t-k$，根据property1，则：

  $PE_{t}^{T}PE_{t+k}=PE_{j}^{T}PE_{j+k}$
  $=PE_{t-k}^{T}PE_{t}$

  其中$d, k, PE^T_jPE_{t+k}$的关系如下图图：

  <img src="https://ybqu.oss-cn-beijing.aliyuncs.com/img/image-20200829163631753.png" style="width: 300px" title=""/>

  上图可以看出，两个位置向量的点积$PE^T_jPE_{t+k}$是对称的，整体上随着$|k|$的增大点积减小，但并不具有单调性，这说明**position embedding不具备方向性**。

---

### 参考：

[【1】NLP三大特征提取器全梳理：RNN vs CNN vs Transformer](https://zhuanlan.zhihu.com/p/189527481?utm_source=wechat_session&utm_medium=social&utm_oi=642818543557677056)

[【2】自然语言处理中的Transformer和BERT](https://zhuanlan.zhihu.com/p/53099098)

[【3】Transformer中的position embedding](https://zhuanlan.zhihu.com/p/166244505)















































