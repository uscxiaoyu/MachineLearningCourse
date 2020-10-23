---
marp: true
# size: 4:3
paginate: true
headingDivider: 0
# header: '**第4章 线性模型**'
---
<!-- fit -->
# 第4章 线性模型

---
# 主要内容

- 线性回归模型
- Logit回归模型
- Softmax回归模型

---
# 0.概述
- 线性回归输出是一个连续值，因此适用于回归问题。回归问题在实际中很常见，如预测房屋价格、气温、销售额等连续值的问题。与回归问题不同，分类问题中模型的最终输出是一个离散值。我们所说的图像分类、垃圾邮件识别、疾病检测等输出为离散值的问题都属于分类问题的范畴。Logit和softmax回归则适用于分类问题。

- 由于线性回归和softmax回归都是单层神经网络，它们涉及的概念和技术同样适用于大多数的深度学习模型。我们首先以线性回归为例，介绍大多数深度学习模型的基本要素和表示方法。

---
# 0.概述
## 模型的基本形式

给定由d个属性描述的示例$\mathbf{x}=(x_1;x_2;...;x_d)$，其中$x_i$是$x$在第i个属性上的取值，线性模型试图学得通过属性的线性组合来进行预测的函数，即
$$f(\mathbf{x})=\omega_1x_1 + \omega_2x_2+...+\omega_dx_d+b,$$
一般用向量形式写出
$$f(\mathbf{x})=\mathbf{\omega^Tx}+b,$$
其中$\mathbf{\omega}=(\omega_1;\omega_2;...;\omega_d)$. $\mathbf{\omega}$和b学得后，模型就确定了.

---
# 1.线性回归模型
![bg right:60% fit](./pictures/4.1.svg)


---
# 1.线性回归模型
- 给定数据集$D=\{(\mathbf{x_1},y_1),(\mathbf{x_2},y_2),...,(\mathbf{x_m},y_m)\}$，其中$\mathbf{x_i}=(x_{i1};x_{i2};...;x_{id}),y_i\in R$. 线性回归试图学得一个线性模型以尽可能准确地预测实数值输出标记. 我们试图学得
$$f(\mathbf{x_i})=\mathbf{x_i}+b,使得f(x_i)\simeq y_i,$$
这称为 **多元线性回归(multivariate linear regression)** .

- 可以利用最小二乘法对$\mathbf{\omega}$和b进行估计。假定增广参数向量$\mathbf{\hat \omega}=(\mathbf{\omega};b)$，相应地
$$
\mathbf{X} =
\begin{pmatrix}
    x_{11} & x_{12} & ... & x_{1d} & 1 \\
    x_{21} & x_{22} & ... & x_{2d} & 1 \\
    ... \\
    x_{m1} & x_{m2} & ... & x_{md} & 1
\end{pmatrix} = 
\begin{pmatrix}
    \mathbf{x_1^T} & 1 \\
    \mathbf{x_2^T} & 1 \\
    ... \\
    \mathbf{x_m^T} & 1
\end{pmatrix}, \mathbf{y}=
\begin{pmatrix}
y_1 \\
y_2 \\
... \\
y_m
\end{pmatrix}=(y_1;y_2;...;y_m)
$$

---
# 1.线性回归模型

## 学习准则
- 如何确定$\mathbf{\hat\omega}$呢？关键在于衡量$f(\mathbf{x})$和$y$的差别。均方误是回归任务中最常用的性能衡量指标，因此我们可以试图让均方误差最小化.
$$
\mathbf{\hat\omega^*}=\operatorname*{argmin}_{\mathbf{\hat\omega}}\mathbf{(y-X\hat\omega)^T(y-X\hat\omega)}.
$$

令$E_{\mathbf{\hat\omega}}=\mathbf{(y-X\hat\omega)^T(y-X\hat\omega)}$，对$\mathbf{\hat\omega}$求导可得
$$
\cfrac{\partial E_{\hat{w}}}{\partial \hat{w}}=2\mathbf{X}^T(\mathbf{X}\hat{w}-\mathbf{y}).
$$

令上式为0可得$\mathbf{\hat\omega}$最优解的封闭解。


---
# 1.线性回归模型
- 当$\mathbf{X^t X}$为满秩矩阵或正定矩阵时，令$\cfrac{\partial E_{\hat{w}}}{\partial \hat{w}}=0$可得
$$
\mathbf{\hat\omega^*=(X^TX)^{-1}X^Ty},
$$

其中$\mathbf{(X^TX)^{-1}}$是$(X^TX)$的逆矩阵. 令$\mathbf{\hat x_i} = (\mathbf{x_i}; 1)$，则最终学得的线性回归模型为
$$
f(\mathrm{\hat x_i})=\mathbf{\hat x_i^T(X^TX)^{-1}X^Ty}.
$$

- 然而，现实任务中$\mathbf{X^t X}$往往不是满秩矩阵，而且随着数据量的增加，计算量呈现大幅增长。因此，往往求助于数值优化算法（如梯度下降）迭代求解。

---
# 1.线性回归模型

## 评价指标

- 回归平方和：$\mathrm{SS_{res}}=\sum_{i=1}^{n}(y_i - f(x_i))^2$

- 总平方和：$\mathrm{SS_{tot}=\sum_{i=1}^n(y_i-\bar{y})^2}$

- 决定系数：$R^2 = \mathrm{\frac{SS_{tot}-SS_{res}}{SS_{tot}}}$


---
# 1.线性回归模型
## 训练方法
- 在求数值解的优化算法中，小批量随机梯度下降(`mini-batch stochastic gradient descent`)在深度学习中被广泛使用。

- 算法过程如下:
    - 先选取一组模型参数的初始值，如随机选取;
    - 接下来对参数进行多次迭代，使每次迭代都可能降低损失函数的值。

> **批量梯度下降法**和**随机梯度下降**可以看作是**小批量随机梯度下降法**的特殊形式，批量梯度下降法使用所有的样本更新参数，随机梯度下降使用1个样本更新参数，小批量随机梯度下降法选择1个小样本更新参数

---
# 1.线性回归模型
## 梯度下降

---
# 1.线性回归模型
```python
def linearModel(X, w):
    return X@w
```


---
# 课堂练习1
请基于`torch`实现多项式回归。
$$
y=\alpha + \sum_{j=1}^{j=p}\beta_j x^j
$$