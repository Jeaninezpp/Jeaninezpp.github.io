---
layout: post
title: Incomplete Multi-view Spectral Clustering with Adaptive Graph Learning 论文
date: 2020-12-05 15:05:00 +0800
category: paper
tags: ["paper","multi-view","incomplete","graph"]
thumbnail: /style/image/paper.jpg
icon: paper
---


* content
{:toc}

## Info
来源：2020 Cybernetics

作者：文杰, Yong Xu∗, Senior Member, IEEE, Hong Liu

缺失多视图论文汇总：[https://github.com/Jeaninezpp/Incomplete-multi-view-clustering](https://github.com/Jeaninezpp/Incomplete-multi-view-clustering)

##  关键词
缺失多视图聚类；图学习

## Introduction
缺失方法可以分为两类：
- **基于矩阵分解的**：
	- 通过矩阵分解得到地位的一致表示。如[（PMVC）](https://blog.csdn.net/zpainter/article/details/106229861)
	- 先填充，然后使用**加权**非负矩阵分解学习一致表示。（MIC），给缺失填充样本较小的权重。
	- **存在问题**：这些基于矩阵分解的方法，共同的问题是它们仅关注于学习一致的表示，而忽略了数据的内在结构，这不能保证所学习到的表示的紧凑性和可辨别性。
- **基于图的**：
	- 旨在学习低维表示。比基于矩阵分解的方法更能有效的探索数据的几何结构。
	- 因此图的创建非常重要，但是因为缺失的原因，无法构造出完整的连接所有样本的图。
		- 先填充。然而缺失比例高时，填充的部分主导表示的学习。
		- [27]使用矩阵分解得到的潜在表示来获得包含全局信息的图。
- **现有方法局限性**
	- 需要有一些样本在所有视图都是完整的。case （a）PMVC/IMG都可以处理。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200612121833177.png)
	- 基于图的方法无法学到全局最优的一致表示。因为子空间学习过程和图构造的过程分离。
- 本文方法：
	- 联合学习低维一致表示和相似图，这样可以获得全局最优的一致表示。
	- 间接的从各个视图的低维表示中学到一个一致的表示。

## 相关工作
1. **谱聚类**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200612121042573.png)
L 最小c个特征值对应的特征向量构成$$n\times c$$的F，作为低维表示。

2. **多视图子空间学习**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200612121353144.png)
Gao el al. 提出MVSC 学到一致的聚类指示矩阵。将图重建和低维表示学习结合在一起，学到一个局部最优的一致表示F，再使用k-means得到最终的聚类结果。

## 方法
- Graph-based 方法存在问题：
	- 每个视图都有缺失，各自构造的图尺寸不一样。	
		- 均值填充。
			- 导致缺失的样本被视为同一类，以同样的权重相连，这使得缺失的样本在低维子空间中被拉在一起，无论他们是否来自同一类。因此这样的图填充方法不合理，尤其是在大比例缺失的情况下。
			- 更合理的方法是**将这些缺失样本相连的权重设置为０**。这样缺失视图中不确定的相似信息将不会在学习数据聚类表示中起负面作用。仅利用可用样本的真实相似性信息来指导表示学习，有利于获得更可靠的数据聚类表示，不免缺失视图的负面影响。
	- 不能反应出所有样本之间的关系。

### 如何将缺失样本权重置为0
$$Z^{(v)}$$是未缺失样本的图，$$\bar{Z}^{(v)}$$是整张图。
通过索引矩阵将$$Z^{(v)}$$缺失样本部分填充为0变成$$\bar{Z}^{(v)}$$
$$\bar{Z}^{(v)}={G^{(v)}}^T Z^{(v)} G^{(v)}$$
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200612125343678.png)
因此$$\bar{L}^{(v)}={G^{(v)}}^T L^{(v)} G^{(v)}$$

得到缺失版的MVSC：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200612125022606.png)
考虑到：
(1) 数据源于低秩的子空间，
(2) 非负的图有利于改善聚类性能并使学到的图更具有解释性。

**添加：
(1) 对图的低秩约束
(2) 对图的非负约束**

### 目标函数
从而目标函数可以写成
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200612130324546.png)

---
其中
$$\sum_{v} \operatorname{Tr}\left(F^{T} G^{(v) T} L^{(v)} G^{(v)} F\right)$$
等价于
$$\frac{1}{2}\sum_{j=1}^{n} \sum_{i=1}^{n}\|F_{i,:}-F_{j,:}\|_{2}^{2} \sum_{v} \bar{W}_{i, j}^{(v)}$$

表明了权重为多个图的相似度之和。
然而在缺失多视图聚类中，这样的方式同等的对待缺失样本的权重和非缺失样本的权重。
这样导致，**属于相同类的样本权重可能会低于不同类的样本权重**。当相同类样本缺失较多，而不同类缺失较少时。

- 为解决这个问题，我们提出从**聚类指示矩阵们**中学习一致的表示。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200612204850467.png)
8式与7式主要不同在于，7式是从多个拉普拉斯矩阵得到一个聚类指示矩阵，而8式是每个拉普拉斯矩阵得到一个拉普拉斯矩阵。
- 使用 $$\Gamma(\cdot)$$ 来度量每个视图的表示 $$F^{(v)}$$与一致表示 $$U$$ 之间的不相似性。除以各自的F范数达到归一化的目的，具有可比性。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200612205049154.png)
K选择线性核，$$K_U=UU^T$$，原因有二。
且
$$\left\|K_{U}\right\|_{F}^{2}=tr(K^TK)=tr(UU^TUU^T)=tr(UU^T)=tr(U^TU)=tr(I_c)=c$$
带入9式中得到
![](https://img-blog.csdnimg.cn/20200612225351298.png)
最终8式可以具体写为下面的**目标函数**：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200612225448376.png)


## 优化
使用ADMM

## 实验
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200612232424100.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3pwYWludGVy,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020061223315963.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3pwYWludGVy,size_16,color_FFFFFF,t_70)


## 缺点
时间复杂度为：$$O\left(\tau\left(k n^{3}+n^{3}+\sum_{v} n_{v}^{3}\right)\right)$$