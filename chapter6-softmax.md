---
marp: true
# size: 4:3
paginate: true
headingDivider: 0
# header: '**第4章 线性模型**'
---
<!-- fit -->
# 第6章 Softmax模型

---
# 主要内容

- Softmax回归模型
- Softmax回归模型的参数学习
- `torch`实现
    - 梯度下降
    - 小批量梯度下降
    - `torch.nn.Module`

---
# 1. Softmax回归模型
- Softmax回归（Softmax Regression），也称为多项（Multinomial）或多类（Multi-Class）的Logit回归，是Logit回归在多分类问题上的推广。

- 对于多类问题，类别标签$y \in {1, 2,..., C}$ 可以有C个取值．给定一个测试样本$x$，Softmax 回归预测的属于类别c的条件概率为
$$
\begin{aligned}
p(y=c|\mathbf{x})&=\mathrm{softmax}(\mathbf{w^T_cx})\\
&=\frac{\exp(\mathbf{w^T_cx})}{\sum_{i=1}^C \exp(\mathbf{w^T_ix})}
\end{aligned}
$$

其中$\mathbf{w_i}$是第i类的权重向量。

---
# 1. Softmax回归模型
```python
def softmax(X, W):
    """
    X: torch.FloatTensor, N*a, N样本数量, a为特征的维度
    W: torch.FloatTensor, a*C, C为类别数量
    """
    C = torch.exp(X@W)  # hat_y, N*C
    # 返回各样本对应类别的标准化概率分布
    return C / torch.sum(C, axis=1).reshape(X.shape[0], -1)
```


---
# 1. Softmax回归模型

- `Softmax`回归的决策函数可以表示为

$$
\begin{aligned}
\hat{y}&=\text{arg}\max_{i=1}^{C}p(y=c|\mathbf{x})\\
&=\text{arg}\max_{i=1}^{C}\mathbf{w_i^Tx}
\end{aligned}
$$

```python
def hat_y(X, W):
    S = softmax(X, W)  # 各样本在各类别上的概率
    max_indices = torch.max(S, dim=1)[1]
    pred_y = torch.zeros_like(S)
    pred_y[torch.arange(S.shape[0]), max_indices] = 1
    return max_indices, pred_y
```

---
# 1. Softmax回归模型

- 与`Logistic`回归的关系。当类别数$C=2$时，`softmax`回归的决策函数为
$$
\begin{aligned}
\hat{y}&=\text{arg}\max_{i\in\{1,2\}}p(y=c|\mathbf{x})\\
&=\text{arg}\max_{i\in\{1,2\}}\mathbf{w_i^Tx}\\
&=I(\mathbf{(w_1-w_0)^Tx}>0)
\end{aligned}
$$
其中$I(\cdot)$是指示函数。



---
# 2. `Softmax`回归模型的参数学习
- 给定N个训练样本，Softmax回归使用交叉熵损失函数学习最有的参数矩阵$W$。为了方便起见，使用C维的`one-hot`向量表示类别标签，对于类别i，其向量表示为
$$
y = [I(i=1), I(i=2), ..., I(i=C)]
$$

- 采用交叉熵损失函数，Softmax回归模型的风险函数是
$$
\begin{aligned}
R(\mathbf{W})&=-\frac{1}{N}\sum_{n=1}^N\sum_{i=1}^{C}y_c^{(n)}\log \hat{y}_c^{(n)}\\
&=-\frac{1}{N}\sum_{n=1}^N(\mathbf{y^{(n)}})^T\log \hat{y}_c^{(n)}
\end{aligned}
$$
- 其中，$\hat{y}_c^{(n)}=\text{softmax}(\mathbf(W^Tx^{(n)}))$为样本$x^{(n)}$在每个类别的后验概率。

---
# 2. `Softmax`回归模型的参数学习

```python
def cross_entropy(X, y, W):
    """
    X: N*(a+1), N个样本, 特征数量为为a, 外加1维偏置
    y: N*C, y为N个C维的one-hot向量
    W: (a+1)*C
    """
    p_y = softmax(X, W) # N*C, N个样本分别在C个类别的后验概率
    # 展开成1维，点积
    crossEnt = -torch.dot(y.reshape(-1), torch.log2(p_y).reshape(-1)) / y.shape[0]
    return crossEnt
```

---
# 2. `Softmax`回归模型的参数学习
- 风险函数$\mathbf{R(W)}$关于$W$的梯度为
$$
\frac{\partial R(W)}{\partial W}=-\frac{1}{N}\sum_{n=1}^N\mathbf{x^{(n)}(y^{(n)}-\hat{y}^{(n)})}^T
$$
