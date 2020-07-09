---
layout: post
title: Graph Embedding on Biomedical Networks
subtitle: Paper reading and summary
date: 2020-07-09
author: Rasin
header-img: img/GNN-1.jpg
catalog: true
tags:
  - Graph Neural Network
  - Biomedical
  - Papers
---

[Graph Embedding on Biomedical Networks: methods, applications and evaluations](https://arxiv.org/abs/1906.05017)

## Abstract

Graph Embedding learning that aims to automatically learn low-dimensional node representations. 

We select 11 representative graph embedding methods and conduct a systematic comparison on 3 important biomedical link prediction tasks: *drug-disease association (DDA) prediction*, *drug-drug interaction (DDI) classification*, *protein-protein interaction (PPI) prediction*; and 2 node classification tasks: *medical term semantic type classification*, *protein function prediction*.

The recent graph embedding methods achieve competitive performance without using any bilogical features and the learned embeddings can be treated as complementary representations for the biological features.

## Introduction

Graphs have been widely used to represent biomedical entites (as nodes) and their relations (as edges).

The goal of graph embedding is to automatically learn a low-dimensional feature representation for each node in the graph. The low-dimensional representations are learned to preserve the structural information of graphs, and thus can be used as features in building machine learning models for various downstream tasks. **Figure 1** summarizes the pipeline for applying various graph embedding methods to downstream prediction tasks.

![](https://raw.githubusercontent.com/rasin-tsukuba/blog-images/master/img/20200708164948.png)

## Overview of graph embedding methods

### MF-based

Matrix Factorization (MF) has been widely adopted for data analyses. Essentially, it aims to factorize a data matrix into lower dimensional matrices and still keep the manifold structure and topological properties hidden in the original data matrix.

Traditional MF has many variants, such as singular value decomposition (SVD) and graph factorization (GF). And they often focus on factorizing the first-order data matrix.

### Random Walk-based

Inspired by the word2vec model, a popular word embedding technique from Natural Language Processing (NLP), which tries to learn word representations from sentences, random walk-based methods are developed to learn node representations by generating **node sequences** through random walks in graphs. Specifically, given a graph and a starting node, random walk-based methods first select one of the node’s neighbors randomly and then move to this neighbor. This procedure is repeated to obtain node sequences. Then the word2vec model is adopted to learn embeddings based on the generated sequences of nodes. In this way, structural and topological information can be preserved into latent features.

DeepWalk, node2vec, and struc2vec are new work in this category.

### Neural Network-based

Various neural networks also have been introduced into graph embedding areas, such as MLP, autoencoder, GAN, and GCN. Different methods adopt different neural architectures and use different kinds of graph infor- mation as input.

- **Line** directly models node embedding vectors by approximating the first-order proximity and second-order proximity of nodes, which can be seen as a single-layer MLP model.
- **DNGR** applies the stacked denoising autoencoders on the positive pointwise mutual information matrix to learn deep low-dimensional node embeddings.
- **SDNE** adopts a deep autoencoder to preserve the second-order proximity by reconstructing the neighborhood strcutre of each node
- **GAE** utilizes a GCN encoder and an inner product decoder to learn node embeddings.
- **GraphGAN** adopts GANs to model the connectivity of nodes. The GAN framework includes a generator and a discriminator where the generator approximates the true connectivity distribution over all other nodes and generates fake samples, while the discriminator model detects whether the sampled nodes are from ground truth or genearted by the generator.

## Applications of Graph Embedding on Biomedical Networks

We select 11 representative graph embedding methods and review how they are used on 3 popular biomedical link prediction applications, and 2 biomedical node classification applications.