---
layout: post
title: Efficient Low-rank Multimodal Fusion with Modality-Specific Factors 论文
date: 2020-12-17 22:00:00 +0800
category: paper
tags: ["paper","multimodal fusion","low-rank", "tensor"]
thumbnail: /style/image/paper.jpg
icon: paper
---

* content
{:toc}
为了解决基于张量的多模态融合方法计算效率差的问题，文章提出了一种低秩多模态融合的方法(Low-rank Multimodal Fusion, LMF)的方法。通过将张量和权重并行分解，利用模态特定的低阶因子来执行多模态融合。避免计算高维的张量，降低了内存开销，将指数级的时间复杂度降低到了线性。

# Introduction
- **融合**的目标是将多种模态结合起来，以利用异质数据的互补性，提供更有力的预测。

- (Fukui et al., 2016), (Zadeh et al., 2017) 使用tensor来进行多模态表示学习。**张量**在多模态表示中具有很大的优越性。但是, 这些方法由于输入tensor的变换，会使得维度会有指数级的增长，计算复杂性也比较高。这严重限制了这些模型的适用性，尤其当数据集有两个以上的模态时。
- 因此这篇文章提出了低秩多模态融合的方法，利用low-rank weight tensors来进行有效的多模态融合。框架如下：
![20201216101757](https://img-blog.csdnimg.cn/img_convert/c05c35749d8153f9f0c41de7fc25e709.png)

- **contributions**
  - 提出低秩多模态融合算法，与模态数呈线性关系。
  - 与SOTA性能相当。
  - 与之前的tensor的方法比，本文提出的方法参数少，效率高。


# 相关工作
- 多模态融合使我们能够利用多模态数据中存在的**互补信息**，从而发现信息对多模态的依赖性。
- 多模态融合方法
  - early fusion: 
    - feature concatenation 直接拼接特征。
    - 直接拼接，甚至有时候会去除时间的依赖，因此 对模态内部(intra-modal)的交互被潜在的抑制，模态内部的上下文信息、时间依赖就会损失
  - late fusion: 
    - 每个模态构造一个模型，然后将输出通过多数表决或者加权平均将结果整合到一起。
    - 但是由于模型是分开创建的，所以视图之间的交互作用不能很好的建模。
  - intermediate: 
    - both intra- and inter- modal.
    - Zadeh et al. (2017) 提出 **Tensor Fusion Network**, 从三个模态计算单个模态表示之间的外积来计算一个张量表示。
但是，这种方法要对多个模态的表示进行外积操作，导致 tensor representation 维度很高，
- 单个模态下 low-rank tensor approximation 应用广泛，但尚未有使用 low-rank tensor 技术来进行多模态融合的。

# Method
文章提出一种模型，将权重分解为低阶因子，这样可以减少模型中参数的数量。这种分解可以通过利用低阶权重张量和输入张量的并行分解来有效地进行基于张量的融合。

## 使用张量表示的多模态融合
这篇论文将多模态融合表述为一个多线性函数 $$f ∶ V_1 × V_2 × … × V_M → H$$。
其中$$\{z_m\}_{m=1}^M$$是M个单个模态的编码信息，而多模态融合的目标是将单模态的表示整合为一个紧凑的多模态表示来进行 下游 的工作。

### tensor fusion
张量表示是一种成功的多模态融合方法，它首先将多输入转换为高维张量，然后将其映射回一个低维输出向量空间。通过对输入模态取外积可以得到张量表示。

为了能够用`一个张量`来模拟任意模态子集之间的相互作用。 Zadeh et al. (2017)提出在进行外积之前，给每个表示$$z$$后面加一个`1`。所以输入的张量$$\mathcal{Z}$$通过单个模态的表示计算得到:$$\mathcal{Z}=\bigotimes_{m=1}^{M} z_{m}, z_{m} \in \mathbb{R}^{d_{m}}$$， $$z_m$$是附加1的输入表示。

输入张量$$\mathcal{Z} \in \mathbb{R}^{d_1,d_2,...,d_m}$$通过一个线性层$$g(\cdot)$$产生一个向量表示：
$$h = g(\mathcal{Z};\mathcal{W},b) = \mathcal{W} ⋅ \mathcal{Z} + b;~h, b \in \mathbb{R}^{d_y}$$
其中$$\mathcal{W}$$是权重，$$b$$是偏移量。

由于$$\mathcal{W}$$是$$M$$阶张量，因此$$\mathcal{W}$$是$$M+1$$阶的张量，维度为$$d_1×d_2×…×d_M×d_h$$，额外的第$$M+1$$层为输出表示的大小$$d_h$$。在进行张量点积的过程中，我们可以把$$\mathcal{W}$$看作是$$d_h$$个$$M$$阶张量，即可以被划分为
$$\overline{\mathcal{W}}_{k} \in \mathbb{R}^{d_{1} \times \ldots \times d_{M}}, k=1, \ldots, d_{h}$$，每一个$$\overline{\mathcal{W}}_{k}$$都在输出的向量$$h$$中贡献一个维度，即$$h_k=\overline{\mathcal{W}}_{k} \cdot \mathcal{Z}$$。

下图为用两个模态的例子来解释**张量融合**：
![20201217145441](https://img-blog.csdnimg.cn/img_convert/dbf8531ca57da082d338c468581f81c7.png)

### drawbacks of tensor fusion
- 我们需要显式地创建一个高维的张量$$\mathcal{Z}$$，其维度为$$\prod_{m=1}^M  d_m$$会随着模态数目呈指数增长。
- 要学习的权重张量 $$\mathcal{W}$$ 也会相应地指数级增长。
- 不仅引入了大量的计算，而且使模型面临着过度拟合的风险。


## 利用模态特定因子进行低秩多模态融合
为了解决tensor-based fusion方法的问题，文章提出了一种低秩多模态融合的方法(Low-rank Multimodal Fusion)(LMF)的方法，将$$\mathcal{W}$$分解为一组modality-specific low-rank factors, 且利用$$\mathcal{Z}$$也可以分解为$$\{z_m\}_{m=1}^M$$。通过这种并行分解的方式，文章可以不显性获得高维的张量而直接计算到$$h$$。

### low-rank weighted decomposition
把$$\mathcal{W}$$看作是$$d_h$$个$$M$$阶张量，每个$$M$$阶张量可以表示为$$\overline{\mathcal{W}}_{k} \in \mathbb{R}^{d_{1} \times \ldots \times d_{M}}, k=1, \ldots, d_{h}$$，存在一个精确分解成向量的模式：$$\overline{\mathcal{W}}_{k}=\sum_{i=1}^{R} \bigotimes_{m=1}^{M} w_{m, k}^{(i)}, ~~~  w_{m, k}^{(i)} \in \mathbb{R}_{m}^{d}$$, 最小的使得分解有效的$$R$$称为张量的rank。

向量的集合$$\left\{\left\{w_{m, k}^{(i)}\right\}_{m=1}^{M}\right\}_{i=1}^{R}$$称为原始张量的秩$$R$$分解因子。

文章固定$$R$$为$$r$$，然后用$$r$$分解因子$$\left\{\left\{w_{m, k}^{(i)}\right\}_{m=1}^{M}\right\}_{i=1}^{r}$$来重建低秩版本的$$\overline{\mathcal{W}}_{k}$$。
这些向量可以重新组合为$$M$$个modality-specific low-rank的因子。令$$\mathbf{w}_{m}^{(i)}=\left[w_{m, 1}^{(i)}, w_{m, 2}^{(i)}, \ldots, w_{m, d_{h}}^{(i)}\right]$$，则模态$$m$$对应的低秩因子为$$\left\{\mathbf{w}_{m}^{(i)}\right\}_{i=1}^{r}$$。

那么低秩的权重张量可以用下式重建得到：$$\mathcal{W}=\sum_{i=1}^{r} \bigotimes_{m=1}^{M} \mathbf{w}_{m}^{(i)}$$

---
基于$$\mathcal{W}$$的分解，再根据$$\mathcal{Z}=\bigotimes_{m=1}^{M} z_{m}$$，我们可以把原来计算$$h$$的式子推算如下：
$$\begin{aligned}
h &=\left(\sum_{i=1}^{r} \bigotimes_{m=1}^{M} \mathbf{w}_{m}^{(i)}\right) \cdot \mathcal{Z} =\sum_{i=1}^{r}\left(\bigotimes_{m=1}^{M} \mathbf{w}_{m}^{(i)} \cdot \mathcal{Z}\right) \\
&=\sum_{i=1}^{r}\left(\bigotimes_{m=1}^{M} \mathbf{w}_{m}^{(i)} \cdot \bigotimes_{m=1}^{M} z_{m}\right) \\
&=\bigwedge_{m=1}^{M}\left[\sum_{i=1}^{r} \mathbf{w}_{m}^{(i)} \cdot z_{m}\right]
\end{aligned}$$

其中$$\bigwedge_{m=1}^{M}$$表示为一系列张量的元素积，即$$\bigwedge_{t=1}^{3} x_{t}=x_{1} \circ x_{2} \circ x_{3}$$。

> - 举一个两模态的例子：
$$\begin{aligned}
h &=\left(\sum_{i=1}^{r} \mathbf{w}_{a}^{(i)} \otimes \mathbf{w}_{v}^{(i)}\right) \cdot \mathcal{Z} 
=\left(\sum_{i=1}^{r} \mathbf{w}_{a}^{(i)} \cdot z_{a}\right) \circ\left(\sum_{i=1}^{r} \mathbf{w}_{v}^{(i)} \cdot z_{v}\right)
\end{aligned}$$

> - 三个模态的流程框架
![20201217215633](https://img-blog.csdnimg.cn/img_convert/7e1a1cad26f4ebd6b146acbcf463da87.png)


这么做的好处显而易见：
- 对$$\mathcal{Z}$$和$$\mathcal{W}$$并行分解，避免了从$$z_m$$去创建高维$$\mathcal{Z}$$的过程。
- 不同的模态之间是解耦的，这使得方法可以扩展到任意模态数目的数据。
- 可微，$$\{\mathbf{w}_{m}^{(i)}\}_{i=1}^r,~m=1,…,M$$可以通过反向传播来优化。
- 将原始的张量融合的方法$$O\left(d_{y} \prod_{m=1}^{M} d_{m}\right)$$的计算复杂性降低到线性$$O\left(d_{y} \times r \times \sum_{m=1}^{M} d_{m}\right)$$。

# Experiment
## Impact of Low-rank Multimodal Fusion
![20201217215851](https://img-blog.csdnimg.cn/img_convert/f6aba13288707bd445d2b1e8e91971a6.png)
实验表明所提方法在所有数据集上都优于Tensor Fusion Network(TFN)

## Complexity Analysis
![20201217220744](https://img-blog.csdnimg.cn/img_convert/e9092ae7df0a9839f3e0508db91873e9.png)
速度超过TFN两倍以上。

## How different low-rank settings impact the performance
![20201217220940](https://img-blog.csdnimg.cn/img_convert/9db676f17aadab75517d6a93711522f1.png)
随着rank的增加，训练结果越来越不稳定，而使用较低的rank就足以达到令人满意的性能了。


# Reference
Amir Zadeh, Minghai Chen, Soujanya Poria, Erik Cam-bria, and Louis-Philippe Morency. 2017. Tensor fu-sion network for multimodal sentiment analysis. In Empirical Methods in Natural Language Processing, EMNLP. 
