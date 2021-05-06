---
layout: post
title: MLP-Mixer An all-MLP Architecture for Vision
date: 2021-05-06
author: Rasin
header-img: img/960x600_paint-liquid-stains-mixing-abstraction-blue.jpg
catalog: true
tags:
  - MLP
  - Computer Vision
  - Neural Network
  - Paper Reading
---

# 简介

虽然卷积神经网络（CNN）已经成为计算机视觉事实上的标准，但最近基于自注意力层的替代品 `Vision Transformers`（ViT）却获得了最先进的性能。`ViT`延续了从模型中消除手工特征和归纳偏差的长期趋势，并进一步依赖于从原始数据中学习。

本文提出了`MLP-Mixer`结构（简称为Mixer），它有竞争力并且在概念和技术上都很简单的替代品，它不适用卷积或自注意力。相反，`Mixer`的结构完全基于多层感知器，这些感知器可以重复应用于空间位置或特征通道。`Mixer`仅依赖于基本矩阵乘法，对数据布局的更改（重塑和换位）以及非线性标量。

`Figure 1`描绘了`Mixer`的宏观结构。它以形状为 `patches x channels`表的一系列线性投影图像补丁（也称为标记）作为输入，并保持该维度。`Mixer`利用两种类型的`MLP`层，`通道混合MLPs`和`符号混合MLPs`。`通道混合MLPs`允许在不同通道之间进行通信，它们独立地对每个符号进行操作，并将表的各个行作为输入。`符号混合MLPs`允许不同空间位置（符号）之间通信；它们在每个通道上独立运行，并以表格的每个列作为输入。这两种类型的层是交错的，以实现两个输入维度的交互。

![Figure 1](https://raw.githubusercontent.com/rasin-tsukuba/blog-images/master/img/20210506140459.png)

极端的说，这种结构可以看作是一个非常特殊的`CNN`，它使用 `1x1`卷积进行通道混合，并使用完整感受野的单通道深度卷积和符号混合的参数共享。此外，卷积比MLP中的普通矩阵乘法更为复杂，因为卷积需要对矩阵乘法和特殊实现实现进行额外的成本。

`Mixer`也具有令人瞩目的成绩。当在大数据集上与训练后（约一亿张图片），它基本上可以达到 SOTA水平。在`ILSVRC2012`上可以达到 `87.94%`第一的验证准确率。在更少量的数据集上训练时（大约一百万到一千万张），加入当代正则化技巧后，`Mixer`也可以去的较好的成绩，与`ViT`媲美，稍弱于特殊的`CNN`架构。

# Mixer 架构

现代深度视觉架构由（1）在给定的空间位置（2）在不同的空间位置之间，或者这两者的组合混合特征的层组成。在 `CNN` 中，第二种层由 \\(N \times N\\)的卷积和池化层组成。在深度网络中，神经元具有巨大的感受野。与此同时，\\(1\times 1\\)的卷积层也可以执行第一种操作，更大的卷积核也可以同时执行以上两种。在`Vision Transformers`或其他基于注意力的架构中，自注意力层允许执行其两者，且`MLP Block`执行其一。`Mixer`架构背后的思想是将按位（通道混合）操作（1）和跨位操作（符号混合）操作（2）清楚地分开。两种操作都由`MLP`来实现。

再来看 `Figure 1`。`Mixer`将不互相重叠的图片区块序列 \\(S\\)作为输入，每个区块都能映射到所需的隐藏维度 \\(C\\)中，这将产生一个二维实值输入表，即 \\(X \in \mathbb{R}^{S \times C}\\)。元时输入图像的分辨率为 \\((H, W)\\)，每个图片区块的大小为 \\((P, P)\\)，那么区块的数量（序列的长度）\\(S = HW / P^2\\)。所有的区块都通过一个相同的映射矩阵线性映射。`Mixer`由相同大小的多层组成，每一层都包括两个 `MLP Blocks`。第一个是`符号混合MLP Block`：它作用于 \\(X\\)的列（即它应用于专职的输入表\\(X^\top\\)），映射\\(\mathbb{R}^S\mapsto\mathbb{R}^S \\)，且与所有的列共享。第二个是`通道混合MLP Block`：它作用于\\(X\\)的行，映射\\(\mathbb{R}^C\mapsto\mathbb{R}^C\\)，与所有的行共享。每个`MLP Block`包含两个全连接层和一个非线性激活函数独立地应用于其输入数据张量的每一行。`Mixer`层可以被写成（忽略层序号）：

\\(\mathbf{U}_{*,i} \ =\ \mathbf{X}_{*,i} +\mathbf{W}_{2} \sigma (\mathbf{W}_{1}\mathrm{LayerNorm}(\mathbf{X})_{*,i}) ,\ \ \mathrm{for\ } i=1\cdots C\\)

\\(\mathbf{Y}_{j,*} \ =\ \mathbf{U}_{j,*} +\mathbf{W}_{4} \sigma (\mathbf{W}_{3}\mathrm{LayerNorm}(\mathbf{U})_{j,*}) ,\ \ \mathrm{for\ } j=1\cdots S\\)

