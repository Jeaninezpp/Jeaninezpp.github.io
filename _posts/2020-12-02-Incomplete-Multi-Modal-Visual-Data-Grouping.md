---
layout: post
title: Incomplete Multi-Modal Visual Data Grouping 论文
date: 2020-12-02 21:00:00 +0800
category: paper
tags: ["paper","multi-view","incomplete"]
thumbnail: /style/image/paper.jpg
icon: paper
---

* content
{:toc}

## Info

文章来源：IJCAI-16

作者：Handong Zhao, Hongfu Liu and Yun Fu

机构：Northeastern University, Boston, USA


## 一句话描述

对通过矩阵分解得到的一致表示应用一个图拉普拉斯项，从而从低维潜在表示中学习一个完整的图

## Introduction

- 缺失多视图方法分类：
  - 删除缺失信息。改变了样本数，违背了原始问题的目标。
  - 填充缺失信息。填充的样本将为视为一类。
- 问题阐述：
  - Partial Multi-View Clustering使用矩阵分解和$$L_1$$范数得到一致潜在子空间。
    ![ ](https://img-blog.csdnimg.cn/20200618101714709.png)
  - Multiple incomplete views clustering via weighted nonnegative matrix factorization with l2,1 regularization. 使用加权NMF和$$L_{2,1}$$范数得到一致潜在表示。
    ![ ](https://img-blog.csdnimg.cn/20200618102528285.png)
- 他们都忽略了整个样本的**全局结构**。
- 本文主要贡献：说明了提出的全局约束在IMG问题中的正确性和有效性。

## Method

- 第一步，对每个视图进行NMF得到潜在表示。
  $$X_C^{(1)}$$,$$X_C^{(2)}$$为两个视图都完整的样本，他们对应得到的表示应该是非常相似的。
  ![ ](https://img-blog.csdnimg.cn/20200618103933807.png)
- 得到的潜在子空间表示 $$P=[P_c;\hat{P}^{(1)},\hat{P}^{(2)}]\in \mathbb{R}^{N\times k}$$,可以直接对其做聚类得到聚类结果，但是这缺乏全局特性，这恰恰是子空间聚类中很重要的问题。
- 提出学习一个在潜在空间中包含所有样本的拉普拉斯矩阵图。
  ![ ](https://img-blog.csdnimg.cn/20200618120325731.png)
  其中$$L_A=D-A$$

### Note

1. 通过拉普拉斯矩阵$$L_A$$，建立起完整视图样本和部分视图样本之间的联系。在目标函数中整合了全局的约束，从而影响地低维空间投影系数的全局结构。
   在完整数据集上添加图项的做法被称作：“complete graph laplacian”
2. A是邻接矩阵，每个元素表示潜在空间中样本之间的相似性。我们将每一列归一化为和为1，且每个元素都是非负的想形式。这样A就具有概率解释。自然的就可以对优化后的A做谱聚类。
3. 公式4中的正则化项，为了简单起见使用了F范数，其他$$L_1$$或者核范数等都是可以用来保存全局结构，有利于聚类性能的提升。

## Optimization

ALM
引入辅助变量Q代替P

## Experiments

- 数据集：
  - synthetic data
  - MSR Action Pairs dataset
  - MSR Daily Activity dataset
  - BUAA NirVis
  - UCI handwritten digit
- 对比算法
  - Best single view
  - Concat
  - MultiNMF
  - PVC
