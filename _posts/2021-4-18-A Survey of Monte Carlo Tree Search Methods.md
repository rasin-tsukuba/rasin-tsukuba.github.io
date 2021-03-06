---
layout: post
title: A Survey of Monte Carlo Tree Search Methods -- Part 1
subtitle: 
date: 2021-04-18
author: Rasin
header-img: img/mcts.png
catalog: true
tags:
  - Games
  - MCTS
  - Paper Reading
---

# 简介

在给定领域，蒙特卡洛搜索树采用在决策空间的随机采样，根据结果来构建一棵搜索树，以找到最优决策。

MCTS具有许多魅力：它是一种即时统计算法，只要有更多的算力就可以带来更好的性能。它基本不需要领域知识，也可以解决许多困难的问题。

## 总览 

MCTS的概念非常简单。这棵搜索树以一种非对称的方式增长。对于算法的每个迭代，`tree policy`被用于寻找当前最需要探索的节点。`Tree policy`企图平衡探索(exploration)与挖掘(exploitation)两方面。之后，从选定的节点开始模拟，根据结果更新树节点。这涉及添加与从选定节点采取的操作相对应的子节点，并更新其祖先的统计信息。 根据一些 `default policy`，一些动作将在模拟的时候执行，最简单的例子就是采用随机动作。MCTS的最大优点在于即时的状态不需要被马上评估（例如深度限制的`minimax`搜索就需要），这样能够极大减少领域的知识量需求。只有在每个模拟最后结束的状态才需要被评估。

虽然基础的MCTS已被证明可解决各种问题，但通常无法实现MCTS的全部优势，除非该基本算法适用于当前领域。大量MCTS研究的重点是确定最适合每种给定情况的那些变化和增强，并了解如何更广泛地使用一个领域的增强。 

# 背景材料

## 决策理论

决策理论将概率论与效用理论相结合，为不确定性下的决策提供了正式而完整的框架。

### 马尔可夫决策过程（MDPs）

马尔可夫决策过程模型序列化的对全可观察的环境进行决策，其中包括四个部分：

* \\(S\\): 一系列状态，\\(s_0\\)为初始状态 
* \\(A\\): 一系列动作
* \\(T(s, a, s')\\): 一个从当前状态\\(s\\)，经由动作 \\(a\\)，到达下一个状态 \\(s'\\) 的转换模型
* \\(R(s)): 奖励函数

总的决策都可以以 \\((s, a)\\) 对来表示，也就是下一个状态 \\(s'\\) 可以由一个当前状态 \\(s\\) 和选定动作 \\(a\\) 的概率分布来决定。策略是一种状态到动作的映射，起目标是使用策略 \\(\pi\\) 来获得最高的期望奖励。

### 部分可见马尔可夫决策过程

如果状态并不是全部可见的，则需要使用部分可见马尔科夫决策过程（POMDP）。这样的话需要增加新的一条：

* \\(O(s, o)\\): 一个观察模型，指定再状态 \\(s\\)中感知概率 \\(o\\)的概率。

大多数优化策略 \\(\pi\\)都是固定的，对于某个状态都可以映射到单个动作而不是一个动作的概率分布。

## 博弈论 

博弈论将决策理论扩展到多个Agent互动的情况。游戏可以定义为一组已建立的规则，该规则允许一个或多个玩家进行互动以产生特定的结果。一个游戏可以由以下部分构成：

* \\(S\\): 一系列状态，\\(s_0\\)为初始状态
* \\(S_T \subseteq S\\): 终止状态集合
* \\(n \in \mathbb{N}\\): 玩家数量 
* \\(A\\): 一系列动作
* \\(f: S \times A \rightarrow S\\): 状态转换函数 
* \\(R: S \rightarrow \mathbb{R}^k\\): 效用函数
* \\(\rho: S \rightarrow (0, 1, \cdots, n) \\): 玩家将在每种状态下行动

每个玩家 \\(k_i\\)执行一个动作，通过函数 \\(f\\)达到下一个状态 \\(s_{t+1}\\)。每个玩家根据他们的动作收到一个奖励（由效用函数 \\(R\\)定义）。这些值可能是随意的，不过一般可能是给赢的 `+1`，平局 `0`分，负的 `-1`。这些是终止状态的博弈论值。

每个玩家的策略都会确定在给定状态\\(s\\)下选择动作 \\(a\\)的概率。如果没有任何玩家能从单方面转换策略中受益，那么玩家策略的组合就会形成纳什均衡。

### 组合游戏

游戏由以下几个属性划分：

* 零和：所有玩家的奖励之和为0
* 信息：游戏状态全部可见或部分可见
* 确定性：是否概率因素起一定作用
* 顺序性：是否动作是顺序执行或同步执行 
* 离散性：动作是离散的或实时的

具有两个成员的游戏，即*零和*，*完全可见信息*，*确定性*，*离散性*和*顺序性*游戏被称为组合游戏。 组合游戏是AI实验的绝佳测试平台，因为它们是由简单规则定义的受控环境，但通常表现出深度和复杂的玩法，可能带来重大的研究挑战，围棋充分证明了这一点。

### 真实游戏中的AI

现实世界中的游戏通常涉及延迟的奖励结构，其中只有在游戏终端状态下获得的那些奖励才能准确地描述每个玩家的表现。 因此，通常将游戏建模为决策树，如下所示： 

* `Minimax` 企图在每个状态下最小化对手的最大奖励，也是二人组合游戏的一种传统搜索方法。通常会过早停止搜索，并使用一个值函数来估算游戏的结果， `α-β` 启发式算法也常用于剪枝。对于非零和游戏和/或具有两个以上玩家的游戏，\\(max^n\\) 算法算法类似于`minimax`。
* `Expectimax` 将minimax概括为随机游戏，在随机游戏中，状态之间的转换是概率性的。 机会节点的值是其子项按其概率加权的总和，否则搜索与\\(max^n\\)相同。 由于机会节点的影响，剪枝策略比较困难。
* `Miximax` 类似于单人 `expectimax`，常用于不完全信息游戏中。它使用了一个预先定义的对手策略来将对手的决策节点当成机会节点。

## 蒙特卡洛方法

这种采样可能有助于近似动作的博弈论值。一个动作的`Q`值表示其期望奖励：

$$
Q(s, a) = \frac{1}{N(s, a)} \sum_{i=1}^{N(s)} \mathbb{I}_i (s, a) z_i
$$

其中，\\(N(s, a)\\)表示的是在状态 \\(s\\)下动作 \\(a\\)被选区的次数，\\(N(s)\\) 为游戏中状态 \\(s\\) 出现的次数， \\(z_i\\) 是状态\\(s\\)中第 \\(i\\)此模拟的结果， 如果在状态\\(s\\)的情况下动作\\(a\\)被选取，\\(\mathbb{I}_i (s, a)\\) 为 `1`，否则为 `0`。

将均匀采样给定状态的动作的蒙托卡罗方法描述为 `flat Monte Carlo`。但是，构造使 `flat Monte Carlo` 故障的退化案例很简单，因为它不允许使用对手模型。通过基于过去的经验来偏向动作选择，可以提高博弈论估计的可靠性。使用到目前为止收集的估计，明智的做法是将动作选择偏向具有较高中间奖励的动作。

## 基于赌博机的方法

赌博机问题是著名的一类顺序决策问题，玩家需要最优选择 \\(K\\) 个动作来最大化累积奖励。由于不知道奖励的分布，选择十分困难，而潜在的奖励都只能基于之前的经验获得。这就导致了 探索-发掘困境：需要平衡对当前被认为是最佳行为的发掘与其他当前看来次优但从长远来看可能是较优的探索行为。

一个 \\(K-\\)臂赌博机被定义为一个随机变量 \\(X_{i, n}\\)，其中\\(1 \leq i \leq K \\) 并且 \\(n \geq 1\\)，\\(i\\)代表着摇杆。接下来的动作都是独立的。可以使用基于过去的奖励来确定要玩哪个摇杆的策略来解决\\(K-\\)臂赌博机问题。 

### 遗憾

策略应旨在最大程度地减少玩家的遗憾，这在n次游玩后定义为： 

$$
R_N = \mu^\star n - \mu_j \sum_{j=1}^{K} \mathbb{E} [T_j(n)]
$$

其中，\\(\mu^\star\\)是最高的可能奖励期望， \\(\mathbb{E} [T_j(n)]\\) 表示了在前 \\(n\\)次尝试后玩摇杆 \\(j\\) 的期望。也就是说，遗憾是指没有选取最佳的摇杆而导致的预期损失。要强调的是，必须始终在所有分支上附加非零概率的必要性，以确保不会因次优分支带来临时有希望的回报而错过最优分支。因此，重要的是对迄今观察到的奖励设置最高置信度界限，以确保做到这一点。

如果遗憾的增长在一定速率的恒定范围内，则随后将其视为解决 探索-发掘 问题的策略。`Upper Confidence Indices`（UCI）可以允许策略估计特定摇杆的预期回报。

### Upper Confidence Bounds (UCB)

UCB1具有一个在\\(n\\)上遗憾的期望对数增长，不需要任何有关奖励分配的先验知识。该策略规定使用摇杆 \\(j\\) 来最大化：

$$
UCB1 = \bar{X}_j + \sqrt{\frac{2 \ln n}{n_j}}
$$

其中， \\(\bar{X}_j\\) 是摇杆 \\(j\\)的平均奖励，\\(n_j\\)是摇杆\\(j\\)被摇动的次数，\\(n\\)是总共游玩的次数。第一项鼓励发掘更高奖励的选择，而第二项鼓励去探索比较少的选择。探究项与平均奖励的单侧置信区间的大小有关，在该区间中，真实预期奖励以压倒性概率落入。