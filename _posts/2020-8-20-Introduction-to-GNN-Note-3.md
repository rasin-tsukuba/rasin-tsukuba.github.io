---
layout: post
title: Introduction to GNN Note 3
date: 2020-08-20
subtitle: Advanced Skills and Applications
author: Rasin
header-img: img/gnn-2-3.jpg
catalog: true
tags:
  - Graph Neural Network
---

## Variants for Advanced Training Methods

### Sampling

GCN requires the full-graph Laplacian, which is computational-consuming for large graphs. Futhermore, GCN is trained independently for a fixed graph, which lacks he ability for inductive learning.

**GraphSAGE** is a comprehensive improvement of the original GCN. It replaced full-graph Laplacian with learnable aggregation functions, which are key to perform message passing and generalize to unseen nodes. They first aggregate neighborhood embeddings, concatenate with target node's embedding, then propagate to the next layer. GraphSAGE could generate embeddings for unseen nodes. It also uses a random neighbor sampling method to alleviate receptive field expansion.

GraphSAGE proposes a way to train the model via batches of nodes instead of the full-graph Laplacian. This enables the training of large graphs though it may be time-consuming.

**PinSAGE** is an extension version of GraphSAGE on large graph. It uses the importance-based sampling method. It defines importance-based neighborhoods of node *u* as the *T* nodes that exert the most influence on node *u*. By simulating random walks starting from target nodes, this approach calculate the L1-normalized visit count of nodes visited by the random walk. Then the top T nodes with the highest normalized visit counts with respect to u are selected to be the neighborhood of node *u*. 

Opposed to node-wise sampling methods that should be performed independently for each node, layer-wise sampling methods that should be performed independently for each node, layer-wise sampling only needs to be performed once. **FastGCN**  further improves the sampling algorithm by interpreting graph convolution as integral transform of embedding functions under probability measure. In stead of sampling neighbors for each node, FastGCN directly samples the receptive field for each layer for variance reduction. It also incorporates importance sampling, in which the importance sampling, in which the importance factor is calculated as below:

$$
q(v) \propto \frac{1}{|N_v|} \sum_{u \in N_v} \frac{1}{|N_v|}
$$

where \\(N_v\\) is the neighborhood of node \\(v\\). The sampling distribution is the same for each layer.

In contrast to fixed sampling methods above, Huang introduce a parameterized and trainable sampler to perform layer-wise sampling. The authors try to learn a self-dependent function \\(g(x(u_j))\\) of each node to determine its importance for sampling based on the node feature \\(x(u_j)\\). The sampling distribution is defined as:

$$
q*(u_j) = \frac{\sum_{i=1}^np(u_j|v_i)|g(x(u_j))|}{sum_{j=1}^N\sum_{i=1}^n p(u_j|v_i)|g(x(v_j))|}
$$

Futhermore, this adaptive sampler could find optimal sampling importance and reduce variance simultaneously.

Many graph analytic problems are solved iteratively and finally achieve steady states. Following the idea of reinforcement learning, **SSE** proposes Stochastic Fixed Point Gradient Descent for GNN training to obtain the same steady-state solutions automatically from examples. This method views embedding update as value function and parameter update as policy function. In training, the algorithm samples nodes to update embeddings and samples labeled nodes to update parameters alternately.

Chen proposed a control-variate based stochastic approximation algorithm for GCN by utilizing the historical activations of nodes as a control variate. This method maintains the historical average activations \\(h_v^{(\hat{l})}\\) to approximate the true activation \\(h_v^{(l)}\\). The advantage of this approach is it limits the receptive field of nodes in the 1-hop neighborhood by using the historical hidden state as an affordable approximation, and the approximation are further proved to have zero variance.

### Hierarchical Pooling

Similar to the pooling layers after a convolutional layer, a lot of work focus on designing hierarchical pooling layers on graphs. Complicated and large-scale graphs usually carry rich hierarchical structures which are of great importance for node-level and graph-level tasks.

Edge-Conditioned Convolution (**ECC**) designs its pooling module with recursively downsampling operation. The downsampling method is based on splitting the graph into two components by the sign of the largest eigenvector of the Laplacian.

**DIFFPOOL** proposes a learnable hierarchical clustering module by training an assignment matrix in each layer:

$$
S^t = softmax(GNN_{pool}^l (A^t, X^t))
$$

where \\(X^t\\) is the matrix of node features and \\(A^t\\) is the coarsened adjacency matrix of layer *t*.

### Data Augmentation

Li focus on the limitations of GCN, which include that GCN requires many additional labeled data for validation and also suffers from the localized nature of the convolutional filter. To solve the limitations, the authors propose Co-Training GCN and Self-Training GCN to enlarge the training dataset. The former method finds the nearest neighbors of training data while the latter one follows a boosting-like way.

## Applications - Structural Scenarios

Structural scenarios mean that data are naturally performed in the graph structure.

### Chemistry and Biology

Molecules and proteins are structured entities that can be represented by graphs. As shown in Figure 12.2, atoms or residues are nodes and chemical bonds or chains are edges. By GNN based representation learning, the learned vectors can help with drug design, chemical reaction
prediction and interaction prediction.

![](https://raw.githubusercontent.com/rasin-tsukuba/blog-images/master/img/20200820110810.png)

#### Molecular Fingerprints

Molecular fingerprints are features vectors representing molecules, which play a key role in computer-aided drug design. Traditional molecular fingerprints rely on heuristic methods which are hand-crafted. GNN provides more flexible approaches for better fingerprints.

The proposed **neural graph fingerprints** which calculate substructure feature vectors via GCN and sum to get overall representation. The aggregation function is

$$
h_{N_v}t = \sum_{u\in N_v} CONCAT(h_u^t, e_{uv})
$$

where \\(e_{uv}\\) is the edge feature of edge (*u, v*). Then update node representation by:

$$
h_v^{t+1} = \sigma (W_t^{deg(v)}h_{N_v}^t)
$$

where \\(deg(v)\\) is the degree of node v and \\(W_t^N\\)  is a learned matrix for each time step t and node degree N.

Kearnes further explicitly model atom and atom pairs independently to emphasize atom interactions. It introduces edge representation \\(e_{uv}^t\\) instead of aggregation function, i.e. \\(h_{N_v}^t=\sum_{u \in N_v} e_{uv}^t\\). The node update function is:

$$
h_v^{t+1} = ReLU(W_1(ReLU(W_0h_u^t), h_{N_v}^t))
$$

while the edge update function is

$$
e_{uv}^{t+1} = ReLU(W_4(ReLU(W_2 e_{uv}^t), ReLU(W_3(h_v^t, h_u^t))))
$$

Beyond atom molecular graphs, some works represent molecules as junction trees. A junction tree is generated by contracting certain vertices in corresponding molecular graph into a single node. The nodes in a junction tree are molecular substructures such as rings and bonds.

Jin leverage variational auto-encoder to generate molecular graphs. Their model follows a two-step process, first generating a junction tree scaffold over chemical substructures, then combining them into a molecule with a graph message passing network.

The proposed VJTNN uses graph attention to decode the junction tree and incorporates GAN for adversarial training to avoid valid graph translation.

To better explain the function of each substructure in a molecule, Lee proposed a game-theoretic approach to exhibit the transparency in structured data. The model is set up as a two-player co-operative game between a predictor and a witness. The predictor is trained to minimize the discrepancy while the goal of the witness is to test how well the predictor conforms to the transparency.

#### Chemical Reaction Prediction

Chemical reaction product prediction is a fundamental problem in organic chemistry. Do view chemical reaction as graph transformation process and introduces GTPN model. GTPN uses GNN to learn representation vectors of reactant and reagent molecules, then leverages reinforcement learning to predict the optimal reaction path in the form of bond change which transforms the reactants into products. 

Bradshaw give another view that chemical reactions can be described as the stepwise redistribution of electrons in molecules. Their model tries to predict the electron paths by learning path distribution over the electron movements. They represent node and graph embeddings with a four-layer GGNN, and then optimize the factorized path generation probability.

#### Medication Recommendation

Using deep learning algorithms to help recommend medications has been explored by researchers and doctors extensively. 

Shang propose GAMENet which takes both longitudinal patient EHR data and drug knowledge based on drug-drug interactions (DDI) as inputs. GAMENet embeds both EHR graph and DDI graph, then feed them into Memory Bank for final output.

To further exploit the hierarchical knowledge for meditation recommendation, Shang combine the power of GNN and BERT for medical code representation. The authors first encode the internal hierarchical structure with GNN, and then feed the embeddings into the pre-trained EHR encoder and the fine-tuned classifier for downstream predictive tasks.

#### Protein and Molecular Interaction Prediction

Fout focus on the task named protein interface prediction, which is a challenging problem to predict the interaction between proteins and the interfaces they occur. The proposed GCN-based method, respectively, learns ligand and receptor protein residue representation and merges them for pairwise classification. Xu introduce MR-GNN which utilizes a multi-resolution model to capture multi-scale node features. The model also utilizes two long short-term memory networks to capture the interaction between two graphs step-by-step.

GNN can also be used in biomedical engineering. With Protein-Protein Interaction Network, Rhee leverage graph convolution and relation network for breast cancer subtype classification. Zitnik also suggest a GCN-based model for polypharmacy side effects prediction Their work models the drug and protein interaction network and separately deals with edges in different types.

## Applications – Other Scenarios

### Generative Models

**NetGAN** is one of the first work to build neural graph generative model, which generates graphs via random walks. It transformed the problem of graph generation to the problem of walk generation which takes the random walks from a specific graph as input and trains a walk generative model using GAN architecture. While the generated graph preserves important topological properties of the original graph, the number of nodes is unable to change in the generating process, which is the same as the original graph. **GraphRNN** manages to generate the adjacency matrix of a graph by generating the adjacency vector of each node step by step, which can output required networks with different numbers of nodes.

Instead of generating adjacency matrix sequentially, **MolGAN** predicts discrete graph structure (the adjacency matrix) at once and utilizes a permutation invariant discriminator to solve the node variant problem in the adjacency matrix. Besides, it applies a reward network for reinforcement learning-based optimization toward desired chemical properties.

Ma propose constrained variational autoencoders to ensure the semantic validity of generated graphs. The authors apply penalty terms to regularize the distributions of the existence and types of nodes and edges simultaneously. The regularization focuses on ghost nodes and valence, connectivity and node compatibility.

**GCPN** incorporated domain-specific rules through reinforcement learning. To successively construct a molecule graph, GCPN follows current policy to decide whether adding an atom or substructure to an existing molecular graph, or adding a bond to connect existing atoms. The model is trained by molecular property reward and adversarial loss collectively.

Li propose a model which generates edges and nodes sequentially and utilize a graph neural network to extract the hidden state of the current graph which is used to decide the action in the next step during the sequential generative process.

Rather than small graphs like molecules,**Graphite** is particularly
suited for large graphs. The model learns a parameterized distribution of adjacent matrix.
Graphite adopts an encoder-decoder architecture, where the encoder is a GNN. For the proposed
decoder, the model constructs an intermediate graph and iteratively refine the graph by
message passing.

Source code generation is an interesting structured prediction task which requires satisfying semantic and syntactic constraints simultaneously. Brockschmidt propose to solve this problem by graph generation. They design a novel model which builds a graph from a partial AST by adding edges encoding attribute relationships. A graph neural network performs message passing on the graph helps better guide the generation procedure.

## Conclusions

Although GNNs have achieved great success in different fields, it is remarkable that GNN models are not good enough to offer satisfying solutions for any graph in any condition. In this section, we will state some open problems for further researches.

### Shallow Structure

Traditional DNNs can stack hundreds of layers to get better performance, because deeper structure has more parameters, which improve the expressive power significantly. However, graph neural networks are always shallow, most of which are no more than three layers.

As experiments in Li show, stacking multiple GCN layers will result in over-smoothing, that is to say, all vertices will converge to the same value. Although some researchers have managed to tackle this problem, it remains to be the biggest limitation of GNN. Designing real deep GNN is an exciting challenge for future research, and will be a considerable contribution to the understanding of GNN.

