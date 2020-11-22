---
marp: true
# size: 4:3
paginate: true
headingDivider: 0
# header: '**第4章 线性模型**'
---
<!-- fit -->
# 第8讲 误差反向传播算法

---
# 主要内容

- 引言
- 误差反向传播的4个方程
- 误差反向传播算法
- 参数初始化和正则化

---
![bg right:90% fit](./pictures/7.back_propagation.svg)

---
# 引言

- 上一节我们看到了神经⽹络如何使⽤梯度下降算法来学习他们⾃⾝的权重和偏置。但是，这⾥还留下了⼀个问题：我们并没有讨论如何计算代价函数的梯度。这是很⼤的缺失！我们接下来学习计算这些梯度的快速算法，也就是反向传播（`backpropagation`）。

- 反向传播算法最初在1970 年代被提及，但是⼈们直到David Rumelhart、Geoffrey Hinton 和Ronald Williams 的著名的1986 年的论⽂中才认识到这个算法的重要性。这篇论⽂描述了对⼀些神经⽹络反向传播要⽐传统的⽅法更快，这使得使⽤神经⽹络来解决之前⽆法完成的问题变得可⾏。

- 现在，反向传播算法已经是神经⽹络学习的重要组成部分了。

---
# 引言


---
# 误差反向传播的4个方程

- 假设输入层表示为$x$, 输入层到第1个隐藏层的权重矩阵表示为$w^1$, 第1个隐藏层的净值为$z^1$, 激活值为$a^1$；输出层的净值为$z^L$, 激活值为$a^L$，即隐藏层加输出层的数量为L。第l层的神经元个数为$M_l$。

- $w_{jk}^l$表示第$(l-1)$层的第$k$个神经元到$l$层的第$j$个神经元的连接上的权重，l-1和l层间的权重矩阵为$w^l\in \mathbb{R}^{\mathbf{M}_l\times \mathbf{M}_{l-1}}$。

- $b_j^l$表示第l-1层到第l层第j个神经元的偏置, l层的偏置向量表示为$b^l\in \mathbb{R}^{\mathbf{M}_l}$。

- $z_j^l$表示第l层第j个神经元的净值, l层的净值向量表示为$z^l\in \mathbb{R}^{\mathbf{M}_l}$。

- $\sigma^l$表示应用于第l层神经元净值的激活函数。

- $a_j^l$表示第l层第j个神经元的激活值, l层的激活值表示为$a^l\in \mathbb{R}^{\mathbf{M}_l}$。

---
![bg right:90% fit](./pictures/7.computational_graph.svg)

---
![bg right:90% fit](./pictures/7.bp3-4.svg)

---
![bg right:90% fit](./pictures/7.error_z.svg)


---
# 参数初始化和正则化

- 参数初始化方法

| 初始化方法 | 激活函数 | 均匀分布$[-r, r]$ | 高斯分布 $N(0, \sigma^2)$ |
| :----: | :----: | :----: | :----: |
| Xavier | Logistic | $r=4\sqrt{\frac{6}{M_{l-1}+M_l}}$ | $\sigma^2=16\times \frac{2}{M_{l-1}+M_l}$ |
| Xavier | tanh | $r=\sqrt{\frac{6}{M_{l-1}+M_l}}$ | $\sigma^2=\frac{2}{M_{l-1}+M_l}$ |
| He | reLu | $r=\sqrt{\frac{6}{M_{l-1}}}$ | $\sigma^2=\frac{2}{M_{l-1}}$ |