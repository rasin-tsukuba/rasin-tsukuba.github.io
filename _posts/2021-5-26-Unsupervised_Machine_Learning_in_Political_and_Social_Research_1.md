---
layout: post
title: Unsupervised Machine Learning in Political and Social Research 1 Introduction
date: 2021-05-26
author: Rasin
header-img: img/ML_in_PSR_1.jpg
catalog: true
tags:
  - Unsupervised Machine Learning
  - Clustering
  - Social Science
---

Original Paper: [Unsupervised Machine Learning for Clustering in Political and Social Research](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3693395)

Header Photo: [From Unsplash](https://unsplash.com/photos/SYTO3xs06fU)

# Introduction

Clustering, which is more aptly situated in *unsupervised machine learning*, allows researchers to explore, learn, and summarize large amounts of data in an effcient way.

Two key components are central to unsupervised machine learning:

1. In unsupervised learning, the researcher works with unlabeled data, meaning classes are not predetermined or specied, and thus there is no expected outcome to be predicted or to predict some other outcome. Unsupervised machine learning works almost exclusively with unlabeled data.
2. Unsupervised learning precludes any meaningful engagement by the researcher during the modeling process. There is no outcome or dependent variable being predicted nor are there parameters to be tuned to result in stronger statistical models used for inference.

Though there is a tradeo between exploration (unsupervised) and confirmation (supervised), it is important to note that each are valuable in their respective spheres, and can even strengthen each other.

When that researcher is more concerned with exploring and summarizing data, perhaps as a step in the broader research program, then unsupervised techniques may be preferred.

Clustering is one of the most common forms of unsupervised machine learning. Clustering algorithms vary widely, and are exceedingly valuable for making sense of often large, unwieldy, unlabeled, and unstructured data by detecting and mapping degrees of similarity between objects in some feature space.

Selecting and fitting clustering algorithms can be complicated for a couple reasons:

1. In clustering there is typically no singl "right" or "one-size-fits-all" algorithm for a question or problem. The selection of an algorithm depends on a variety of factors such as: 
  * the size and structure of the data
  * the goals of the researcher
  * the level of domain expertise
  * the transformation of the data
  * how observations are treated
2. In unsupervised learning there are no parameters to be estimated nor are there clear expectations for emergent patterns as there are in supervised learning. As a result, evaluation of unsupervised algorithmic performance is rarely straightforward.

## Running Example: State Legislative Professionalism

We use state legislative professionalism as a running example.

The state legislative professionalism data include 950 observations (states) from 1974 to 2010, where each row is a state/year dyad, e.g., `Alabama, 1984.`

There are four key inputs of interest: 
  * `total session length`: range from `36` days to `549.54` days.
  * `regular session length`: range from `36` days to `521.85` days.
  * `salaries of legislators`: range from `0` to `$254.93`.
  * `expenditures per legislator`: range from `$40.14` to `$5523.10`.

In sum, these data include rich nuance from states with large budgets and well paid legislators to states with volunteer legislatures and small budgets, making them valuable for such an exercise in unsupervised clustering.

## Visualization as a Key Tool

As data visualization can be understood as perhaps the simplest form of unsupervised learning.

## A Word on Programming in R

All examples and code used throughout are executed in the R programming language.

The following packages will be used throughout this Element. Readers are encouraged to install all packages first, and then load the libraries using the following code.

```
# first install the packages
install.packages(c(" tidyverse ", " factoextra ", " skimr ", "ape", " clValid ", " cluster ", " gridExtra ", " mixtools "))

# next , load the libraries
library ( tidyverse )
library ( factoextra )
library ( skimr )
library (ape )
library ( clValid )
library ( cluster )
library ( gridExtra )
library ( mixtools )
```
