## BERT（Bidirectional Encoder Representations from Transformers）模型 （.vs GPT）

[TOC]

---

### 1. BERT vs GPT

#### 1.1 模型结构

<img src="https://ybqu.oss-cn-beijing.aliyuncs.com/img/image-20200829192406784.png" style="width: 200px" title=""/>

+ GPT的模型结构和BERT是相同的，都是上图的结构，只是BERT的模型规模更加庞大；

#### 1.2 模型预训练

+ **GPT预训练：**在一个8亿单词的语料库上做训练，给出前文，不断地预测下一个单词。比如Winter is coming，当给出第一个词Winter之后，预测下一个词is，之后再预测下一个词coming。不需要标注数据，通过这种无监督训练的方式，得到一个预训练模型；

+ **BERT预训练：**在一个33亿单词的语料库上做预训。预训练包括了两个任务：

  + 第一个任务是随机地扣掉15%的单词，用一个掩码MASK代替，让模型去猜测这个单词；

    + 80% 的时间：用[MASK]替换目标单词，例如：my dog is hairy --> my dog is [MASK] ；
    + 10% 的时间：用随机的单词替换目标单词，例如：my dog is hairy --> my dog is apple ；
    + 10% 的时间：不改变目标单词，例如：my dog is hairy --> my dog is hairy 。 （这样做的目的是使表征偏向于实际观察到的单词。）

  + 第二个任务是，每个训练样本是一个上下句，有50%的样本，下句和上句是真实的，另外50%的样本，下句和上句是无关的，模型需要判断两句的关系。这两个任务各有一个loss，将这两个loss加起来作为总的loss进行优化。

    **正样本：我[MASK]（是）个算法工程师，我服务于WiFi万能钥匙这家[MASK]（公司）。**

    **负样本：我[MASK]（是）个算法工程师，今天[MASK]（股票）又跌了。**

+ **GPT和BERT两种预训练方式对比：**

  + GPT在预测词的时候，只预测下一个词，因此只能用到上文的信息，无法利用到下文的信息；
  + BERT是预测文中扣掉的词，可以充分利用到上下文的信息，这使得模型有更强的表达能力，这也是BERT中**Bidirectional**的含义。在一些NLP任务中需要判断句子关系，比如判断两句话是否有相同的含义。BERT有了第二个任务，就能够很好的捕捉句子之间的关系。下图是BERT原文中对另外两种方法的预训练对比，包括GPT和ELMo。水平方向的Transformer表示的是同一个单元，图中复杂的连线表示的是词与词之间的依赖关系，BERT中的依赖关系既有前文又有后文，而GPT的依赖关系只有前文。

  <img src="https://ybqu.oss-cn-beijing.aliyuncs.com/img/image-20200829193746343.png" style="width: 700px" title=""/>

  > ELMo是基于feature-based$^{[1]}$的方法应用pre-trained language representations，分别使用了left-to-right和right-to-left进行独立训练，然后将输出拼接起来，为下游任务提供序列特征;
  >
  > BERT和OpenAI GPT是基于fine-tuning$^{[2]}$的方法应用pre-trained language representations。BERT使用双向的Transformer架构，OpenAI GPT使用了left-to-right的Transformer。

---

### 2. BERT模型

#### 2.1 pre-training 和 fine-tuning

使用BERT有2个步骤：

+ **pre-training：** 预训练期间，BERT模型在不同任务的未标记数据上进行训练（无监督学习）；
+ **fine-tuning：**微调的时候，BERT模型用预训练好的参数进行初始化，并且是基于下游任务的有标签的数据来训练的。每个下游任务有自己的微调模型，尽管最初的时候都是用的预训练好的BERT模型参数。

<img src="https://ybqu.oss-cn-beijing.aliyuncs.com/img/image-20200829195327269.png" style="width: 700px" title=""/>

上图是BERT的pre-training和fine-tuning运行过程。除了output层，这两个阶段的架构是一样的。预训练模型的参数会做为不同下游任务的模型的初始化参数。在fine-tuning时，所有参数参与微调。[CLS]是一个特别设置的符号，添加在每个输入样本的前面，表示这是一个输入样本的开始，[SEP]是特别设置的一个分隔标记。比如分隔questions/answers。

#### 2.2 模型架构

+ BERT的模型架构是一个多层双向Transformer编码器。

+ 两种模型：
  + BERT(base，L=12, H=768, A=12, Total Parameters=110M)；
  + BERT(large，L=24, H=1024, A=16, Total Parameters=340M)；
  + **注**：$L$表示层数，$H$表示每个隐藏单元的维数大小，$A$表示self-attention头数。
+ 使用BERT做各种下游任务，输入表征可以在一个token序列里清楚的表示一个句子或者一对句子(比如<Question,Answer>)。
  + “句子”不是必须是语言句子，而可以是任意范围的连续文本；
  + “sequence”指BERT的输入序列，可以是一个句子，也可以是两个打包在一起的句子。
+ 使用了WordPiece embeddings来做词嵌入，对应的词汇表有30000个token。每个序列的首个token总是一个特定的classification token([CLS])。这个token对应的最后的隐藏状态被用作分类任务的聚合序列表征。句子对打包成一个序列。有两种区分句子对中的句子的方法：
  + 第一种，通过分隔符[SEP]；
  + 第二种，模型架构中添加了一个经过学习的嵌入(learned embedding)到每个token，以表示它是属于句子A或者句子B。如上图中，$E$表示输入的词嵌入，$C$表示最后隐藏层的[CLS]的向量，$T_i$表示第$i$个输入token在最后隐藏层的向量。

+ 对一个给定的token，其输入表征由对应的token，segment和position embeddings的相加来构造。如下图：

![](https://ybqu.oss-cn-beijing.aliyuncs.com/img/image-20200829200916498.png)

#### 2.3 具体任务微调

<img src="https://ybqu.oss-cn-beijing.aliyuncs.com/img/image-20200829201645170.png" style="width: 600px" title=""/>

+ **分类任务**

  + 多句子分类任务，比如判断两句话是否表示相同的含义。如上图（a）；

  + 单句子分类任务，比如判断电影评论是喜欢还是讨厌。如上图（b）；
    $$
    P=softmax(C\cdot W^T)
    $$

  + 在输出的隐向量中，取出CLS对应的向量C，加一层网络W，并丢给softmax进行分类，得到预测结果P，计算过程如公式（1）。在特定任务数据集中对Transformer模型的所有参数和网络W共同训练，直到收敛。新增加的网络W是$H\times K$维，H表示隐向量的维度，K表示分类数量，W的参数数量相比预训练模型的参数少得可怜；

+ **问答任务**

  + 如上图（c），以SQuAD v1.1为例，给出一个问题Question，并且给出一个段落Paragraph，然后从段落中标出答案的具体位置。需要学习一个开始向量S，维度和输出隐向量维度相同，然后和所有的隐向量做点积，取值最大的词作为开始位置；另外再学一个结束向量E，做同样的运算，得到结束位置。
    + **注：**结束位置一定要大于开始位置。

+ **NER任务**

  + 比如给出一句话，对每个词进行标注，判断属于人名，地名，机构名，还是其他。如图（d）所示，加一层分类网络，对每个输出隐向量都做一次判断。

---

### 参考

[【1】BERT论文解读](https://www.cnblogs.com/anai/p/11645953.html)

[【2】自然语言处理中的Transformer和BERT](https://zhuanlan.zhihu.com/p/53099098)

---

[^1]:feature-based，又称feature-extraction 特征提取。就是用预训练好的网络在新样本上提取出相关的特征，然后将这些特征输入一个新的分类器，从头开始训练的过程。也就是说在训练的过程中，网络的特征提取层是被冻结的，只有后面的密集链接分类器部分是可以参与训练的。
[^2]:fine-tuning，微调。和feature-based的区别是，训练好新的分类器后，还要解冻特征提取层的顶部的几层，然后和分类器再次进行联合训练。之所以称为微调，就是因为在预训练好的参数上进行训练更新的参数，比预训练好的参数的变化相对小，这个相对是指相对于不采用预训练模型参数来初始化下游任务的模型参数的情况。也有一种情况，如果你有大量的数据样本可以训练，那么就可以解冻所有的特征提取层，全部的参数都参与训练，但由于是基于预训练的模型参数，所以仍然比随机初始化的方式训练全部的参数要快的多。

