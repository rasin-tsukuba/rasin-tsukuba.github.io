---
layout: post
title: NLP with PyTorch 2 Quick Tour of Traditional NLP
subtitle: 
date: 2020-07-22
author: Rasin
header-img: img/nlp-2.jpg
catalog: true
tags:
  - Deep Learning
  - Natural Language Processing
  - Tutorials
---

[Chapter 2.A Quick Tour of Traditional NLP](https://yifdu.github.io/2018/12/18/Natural-Language-Processing-with-PyTorch%EF%BC%88%E4%BA%8C%EF%BC%89/)

## Conceptions

自然语言处理(NLP)和计算语言学(CL)是人类语言计算研究的两个领域。NLP旨在开发解决涉及语言的实际问题的方法，如信息提取、自动语音识别、机器翻译、情绪分析、问答和总结。另一方面，CL使用计算方法来理解人类语言的特性。

### Corpora, Tokens, and Types

所有的NLP方法，无论是经典的还是现代的，都以文本数据集开始，也称为语料库(Corpora)。语料库通常有原始文本和与文本相关的任何元数据。原始文本是字符(字节)序列，但是大多数时候将字符分组成连续的称为令牌(Tokens)的连续单元。在英语中，令牌(Tokens)对应由空格字符或标点分隔的单词和数字序列。元数据可以是与文本相关联的任何辅助信息，例如标识符，标签和时间戳。

在机器学习术语中，文本及其元数据称为实例或数据点。 

如下图所示，语料库是一组实例，也称为数据集。

![](https://raw.githubusercontent.com/rasin-tsukuba/blog-images/master/img/20200722082346.png)

将文本分解为令牌(Tokens)的过程称为令牌化(tokenization)。世界语的句子：`"Maria frapis la verda sorĉistino"`有六个令牌。

令牌化可能比简单地基于非字母数字字符拆分文本更加复杂。如下图所示，对于像土耳其语这样的粘合语言来说，分隔空格和标点符号可能是不够的，因此可能需要更专业的技术。

![](https://raw.githubusercontent.com/rasin-tsukuba/blog-images/master/img/20200722082558.png)

如下面这条推文：

![](https://raw.githubusercontent.com/rasin-tsukuba/blog-images/master/img/20200722082719.png)

令牌化推特涉及到保存话题标签和特殊内容，将表情(如`:-)`)和urls分割为一个单元。`#MakeAMovieCold`标签应该是1个令牌还是4个?虽然大多数研究论文对这一问题并没有给予太多的关注，而且事实上，许多令牌化决策往往是任意的，但是这些决策在实践中对准确性的影响要比公认的要大得多。这通常被认为是预处理的繁琐工作，大多数开放源码NLP包为令牌化提供了合理的支持。

示例2-1展示了来自NLTK和SpaCy的示例，这是两个用于文本处理的常用包。

**SpaCy样例**

```
import spacy
nlp = spacy.load(‘en’)
text = “Mary, don’t slap the green witch”
print([str(token) for token in nlp(text.lower())])
```

输出：

```
['mary', ',', 'do', "n't", 'slap', 'the', 'green', 'witch', '.']
```

**NLTK样例**

```
from nltk.tokenize import TweetTokenizer
tweet=u"Snow White and the Seven Degrees
    #MakeAMovieCold@midnight:-)"
tokenizer = TweetTokenizer()
print(tokenizer.tokenize(tweet.lower()))
```

输出：

```
['snow', 'white', 'and', 'the', 'seven', 'degrees', '#makeamoviecold', '@midnight', ':-)']
```

类型是语料库中唯一的令牌。语料库中所有类型的集合就是它的词汇表或词典。词可以区分为内容词和停用词。像冠词和介词这样的限定词主要是为了达到语法目的，就像填充物承载着内容词一样。

### Ngrams

ngram是文本中出现的固定长度(n)的连续令牌序列。bigram有两个令牌，unigram 只有一个令牌。SpaCy和NLTK等包提供了方便的方法

```
def n_grams(text, n):
    '''
    takes tokens or text, returns a list of n grams
    '''
    return [text[i:i+n] for i in range(len(text)-n+1)]

cleaned = ['mary', ',', "n't", 'slap', green', 'witch', '.']
print(n_grams(cleaned, 3))

```

输出：

```
[['mary', ',', "n't"],
 [',', "n't", 'slap'],
 ["n't", 'slap', 'green'],
 ['slap', 'green', 'witch'],
 ['green', 'witch', '.']]
```

对于子词(subword)信息本身携带有用信息的某些情况，可能需要生成字符ngram。例如，`“methanol”`中的后缀`“-ol”`表示它是一种醇;如果您的任务涉及到对有机化合物名称进行分类，那么您可以看到ngram捕获的子单词(subword)信息是如何有用的。在这种情况下，您可以重用相同的代码，将每个字符ngram视为令牌。(这里的subword应该是值类似前缀后缀这种完整单词中的一部分)

### Lemmas and Stems

Lemmas是单词的词根形式。考虑动词`fly`。它可以被屈折成许多不同的单词——`flow`、`fly`、`flies`、`flying`、等等——而`fly`是所有这些看似不同的单词的Lemmas。有时，为了保持向量表示的维数较低，将令牌减少到它们的Lemmas可能是有用的。这种简化称为lemmatization。

在以下例子中，SpaCy使用一个预定义的字典WordNet来提取Lemmas，但是lemmatization可以构建为一个机器学习问题，需要理解语言的形态学。

```
import spacy
nlp = spacy.load(‘en’)
doc = nlp(u"he was running late")
for token in doc:
    print('{} --> {}'.format(token, token.lemma_))
```

Stem词干是最普通的lemmatization。它涉及到使用手工制定的规则来去掉单词的结尾，从而将它们简化为一种叫做词干的常见形式。通常在开源包中实现的流行的词干分析器是`Porter`的`Stemmer`和`Snowball Stemmer`。

## Categorizing Sentences and Documents

对文档进行归类或分类可能是NLP最早的应用之一。主题标签的分配、评论情绪的预测、垃圾邮件的过滤、语言识别和邮件分类等问题可以被定义为受监督的文档分类问题。

### Categorizing Words: POS Tagging

我们可以将标记的概念从文档扩展到单个单词或标记。分类单词的一个常见示例是词性标注，如下例所示：

```
import spacy
nlp = spacy.load(‘en’)
doc = nlp(u"Mary slapped the green witch.")
for token in doc:
    print('{} - {}'.format(token, token.pos_))
```

输出：

```
Mary - PROPN
slapped - VERB
the - DET
green - ADJ
witch - NOUN
. - PUNCT
```

### Categorizing Spans: Chunking and Named Entity Recognition

通常，我们需要标记文本的范围，即一个连续的多令牌边界。例如，`“Mary slapped the green witch.”`我们可能需要识别其中的名词短语(NP)和动词短语(VP)，如下图所示:

![](https://raw.githubusercontent.com/rasin-tsukuba/blog-images/master/img/20200722091234.png)

这称为分块(Chunking)或浅解析(Shallow parsing)。浅解析的目的是推导出由名词、动词、形容词等语法原子组成的高阶单位。如果没有训练浅解析模型的数据，可以在词性标记上编写正则表达式来近似浅解析。

一下示例给出了一个使用SpaCy的浅解析示例，输出名词短语。

```
import spacy
nlp = spacy.load(‘en’)
doc  = nlp(u"Mary slapped the green witch.")
for chunk in doc.noun_chunks:
    print '{} - {}'.format(chunk, chunk.label_)
```

输出：

```
Mary - NP
the green witch - NP
```

另一种有用的span类型是命名实体。命名实体是一个字符串，它提到了一个真实世界的概念，如人员、位置、组织、药品名称等等。这里有一个例子:

![](https://raw.githubusercontent.com/rasin-tsukuba/blog-images/master/img/20200722091654.png)

### Structure of Sentences

浅层解析识别短语单位，而识别它们之间关系的任务称为解析(parsing)。解析树(Parse tree)表示句子中不同的语法单元在层次上是如何相关的。我们对 `"Mary slapped the green witch."` 这句话进行成分分析：

![](https://raw.githubusercontent.com/rasin-tsukuba/blog-images/master/img/20200722091947.png)

另一种可能更有用的显示关系的方法是使用依赖项解析(dependency parsing)：

![](https://raw.githubusercontent.com/rasin-tsukuba/blog-images/master/img/20200722092053.png)

### Word Senses and Semantics

单词有意义，而且通常不止一个。一个词的不同含义称为它的意义(senses)。WordNet是一个长期运行的词汇资源项目，它来自普林斯顿大学，旨在对所有英语单词的含义以及其他词汇关系进行分类。

下图展示单词 `plane` 一词的不同用法：

![](https://raw.githubusercontent.com/rasin-tsukuba/blog-images/master/img/20200722092159.png)

在WordNet这样的项目中数十年的努力是值得的，即使是在有现代方法的情况下。词的意义也可以从上下文中归纳出来。从文本中自动发现词义实际上是半监督学习在自然语言处理中的第一个应用。

## Summary

在这一章中，我们回顾了NLP中的一些基本术语和思想，这些在以后的章节中会很有用。本章只涉及了传统NLP所能提供的部分内容。在许多情况下，基于神经网络的方法应该被看作是传统方法的补充而不是替代。有经验的实践者经常使用这两个世界的优点来构建最先进的系统。