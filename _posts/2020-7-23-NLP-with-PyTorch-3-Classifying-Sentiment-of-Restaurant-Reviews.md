<<<<<<< HEAD
---
layout: post
title: NLP with PyTorch 3 Classifying Sentiment of Restaurant Reviews
subtitle: 
date: 2020-07-23
author: Rasin
header-img: img/nlp-31.jpg
catalog: true
tags:
  - Deep Learning
  - Natural Language Processing
  - Tutorials
---

[Chapter 3. Classifying Sentiment of Restaurant Reviews](https://yifdu.github.io/2018/12/19/Natural-Language-Processing-with-PyTorch%EF%BC%88%E4%B8%89%EF%BC%89/)

## Prologue

在上一节中，我们通过一个玩具示例深入研究了有监督的训练，并阐述了许多基本概念。在本节中，我们将重复上述练习，但这次使用的是一个真实的任务和数据集：使用感知器和监督培训对Yelp上的餐馆评论进行分类，判断它们是正面的还是负面的。

在这个例子中，我们使用Yelp数据集，它将评论与它们的情感标签(正面或负面)配对。此外，我们还描述了一些数据集操作步骤，这些步骤用于清理数据集并将其划分为训练、验证和测试集。

在理解数据集之后，您将看到定义三个辅助类的模式，这三个类在本书中反复出现，用于将文本数据转换为向量化的形式:词汇表(Vocabulary)、向量化器(Vectorizer)和PyTorch的`DataLoader`:
- 词汇表协调我们在“观察和目标编码”中讨论的整数到令牌(token)映射。我们使用一个词汇表将文本标记(text tokens)映射到整数，并将类标签映射到整数。
- 接下来，向量化器(vectorizer)封装词汇表，并负责接收字符串数据，如审阅文本，并将其转换为将在训练例程中使用的数字向量。
- 我们使用最后一个辅助类，PyTorch的DataLoader，将单个向量化数据点分组并整理成minibatches。

## The Yelp Review Dataset

2015年，Yelp举办了一场竞赛，要求参与者根据点评预测一家餐厅的评级。同年，Zhang, Zhao，和Lecun(2015)将1星和2星评级转换为“消极”情绪类，将3星和4星评级转换为“积极”情绪类，从而简化了数据集。该数据集分为56万个训练样本和3.8万个测试样本。在这个数据集部分的其余部分中，我们将描述最小化清理数据并导出最终数据集的过程。然后，我们概述了利用PyTorch的数据集类的实现。

在这个例子中，我们使用了简化的Yelp数据集，但是有两个细微的区别。第一个区别是我们使用数据集的“轻量级”版本，它是通过选择10%的训练样本作为完整数据集而派生出来的。这导致了两种结果：
- 首先，使用一个小数据集可以使训练测试循环快速，因此我们可以快速地进行实验。
- 其次，它生成的模型精度低于使用所有数据。这种低精度通常不是主要问题，因为您可以使用从较小数据集子集中获得的知识对整个数据集进行重新训练。在训练深度学习模型时，这是一个非常有用的技巧，因为在许多情况下，训练数据的数量是巨大的。

=======
---
layout: post
title: NLP with PyTorch 3 Classifying Sentiment of Restaurant Reviews
subtitle: 
date: 2020-07-23
author: Rasin
header-img: img/nlp-31.jpg
catalog: true
tags:
  - Deep Learning
  - Natural Language Processing
  - Tutorials
---

[Chapter 3. Classifying Sentiment of Restaurant Reviews](https://yifdu.github.io/2018/12/19/Natural-Language-Processing-with-PyTorch%EF%BC%88%E4%B8%89%EF%BC%89/)

## Prologue

在上一节中，我们通过一个玩具示例深入研究了有监督的训练，并阐述了许多基本概念。在本节中，我们将重复上述练习，但这次使用的是一个真实的任务和数据集：使用感知器和监督培训对Yelp上的餐馆评论进行分类，判断它们是正面的还是负面的。

在这个例子中，我们使用Yelp数据集，它将评论与它们的情感标签(正面或负面)配对。此外，我们还描述了一些数据集操作步骤，这些步骤用于清理数据集并将其划分为训练、验证和测试集。

在理解数据集之后，您将看到定义三个辅助类的模式，这三个类在本书中反复出现，用于将文本数据转换为向量化的形式:词汇表(Vocabulary)、向量化器(Vectorizer)和PyTorch的`DataLoader`:
- 词汇表协调我们在“观察和目标编码”中讨论的整数到令牌(token)映射。我们使用一个词汇表将文本标记(text tokens)映射到整数，并将类标签映射到整数。
- 接下来，向量化器(vectorizer)封装词汇表，并负责接收字符串数据，如审阅文本，并将其转换为将在训练例程中使用的数字向量。
- 我们使用最后一个辅助类，PyTorch的DataLoader，将单个向量化数据点分组并整理成minibatches。

## The Yelp Review Dataset

2015年，Yelp举办了一场竞赛，要求参与者根据点评预测一家餐厅的评级。同年，Zhang, Zhao，和Lecun(2015)将1星和2星评级转换为“消极”情绪类，将3星和4星评级转换为“积极”情绪类，从而简化了数据集。该数据集分为56万个训练样本和3.8万个测试样本。在这个数据集部分的其余部分中，我们将描述最小化清理数据并导出最终数据集的过程。然后，我们概述了利用PyTorch的数据集类的实现。

在这个例子中，我们使用了简化的Yelp数据集，但是有两个细微的区别。第一个区别是我们使用数据集的“轻量级”版本，它是通过选择10%的训练样本作为完整数据集而派生出来的。这导致了两种结果：
- 首先，使用一个小数据集可以使训练测试循环快速，因此我们可以快速地进行实验。
- 其次，它生成的模型精度低于使用所有数据。这种低精度通常不是主要问题，因为您可以使用从较小数据集子集中获得的知识对整个数据集进行重新训练。在训练深度学习模型时，这是一个非常有用的技巧，因为在许多情况下，训练数据的数量是巨大的。

>>>>>>> master
从这个较小的子集中，我们将数据集分成三个分区:一个用于训练，一个用于验证，一个用于测试。虽然原始数据集只有两个部分，但是有一个验证集是很重要的。在机器学习中，您经常在数据集的训练部分上训练模型，并且需要一个held-out部分来评估模型的性能。如果模型决策基于held-out部分，那么模型现在不可避免地偏向于更好地执行held-out部分。因为度量增量进度是至关重要的，所以这个问题的解决方案是使用第三个部分，它尽可能少地用于评估。