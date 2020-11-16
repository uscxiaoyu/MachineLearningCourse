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
![bg right:50% fit](./pictures/5.2-sofmax.svg)

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

```python
def grad_crosEnt_W(X, y, W):
    '''
    X: N*(a+1), N个样本, 特征数量为为a, 外加1维偏置
    y: N*C, y为N个C维的one-hot向量
    W: (a+1)*C
    '''
    hat_y = softmax(X, W)
    a = (X.t() @ (y - hat_y)) / y.shape[0]  # (a+1)*N | N*C
    return a
```

---
## 2. `Softmax`回归模型的参数学习

**采用梯度下降法，softmax回归的训练过程为**

- 输入: 训练集X，`one-hot`形式的标签y
- 输出: 最优参数$w^*$
- 算法过程
    - 初始化$W_0:=0$，最大迭代次数$T$
    - 然后通过下式进行参数的迭代更新
    $$
    W_{t+1}:=W_t+\eta\left(\frac{1}{N}\sum_{n=1}^N\mathbf{x^{(n)}(y^{(n)}-\hat{y}^{(n)})}^T\right) 
    $$
    - 直到满足指定迭代次数，令$w^*=w^T$。
                                                                                                                                                
---
## 3. `Softmax`回归模型的参数学习
- 方法1: 梯度下降-人工求导

```python
def softmax_sgd(X, y, num_steps=100, lr=0.1):
    '''
    X: N*(a+1), N个样本, 特征数量为为a, 外加1维偏置
    y: N*C, y为N个C维的one-hot向量
    W: (a+1)*C
    '''
    hat_X = torch.cat([X, torch.ones(X.shape[0], 1)], axis=1)  # 增广X
    W = torch.randn(hat_X.shape[1], y.shape[1])  # 增广参数矩阵
    for i in range(num_steps):
        W += lr*grad_crosEnt_W(hat_X, y, W)
        loss = cross_entropy(hat_X, y, W)
        if (i+1) % 50 == 0:
            print(f'训练{i+1}轮, 交叉熵为{loss:.2f}')
            
    return W
```


---
## 3. `Softmax`回归模型的参数学习
- 方法2: 随机梯度下降-自动求导

```python
def softmax_miniBatch_sgd(X, y, num_epoch=50, batch_size=40, lr=0.05):
    '''
    X: N*a, N个样本, 特征数量为为a
    y: N*C, y为N个C维的one-hot向量
    W: a*C
    '''
    hat_X = torch.cat([X, torch.ones(X.shape[0], 1)], axis=1)  # 增广X
    W = torch.randn(hat_X.shape[1], y.shape[1])  # 增广参数矩阵
    W.requires_grad_()
    dataset = TensorDataset(hat_X, y)
    data_iter = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(num_epoch):
        for t_x, t_y in data_iter:
            l = cross_entropy(t_x, t_y, W)        
            l.backward()  # 计算损失函数在 W 上的梯度
            W.data.sub_(lr*W.grad/batch_size)
            W.grad.data.zero_()
            
        if (epoch + 1) % 50 == 0:
            with torch.no_grad():  # 不计算梯度，加速损失函数的运算
                train_l = cross_entropy(hat_X, y, W)  # 最近一次的负对数似然率
                est_W = W.detach().numpy()  # detach得到一个有着和原tensor相同数据的tensor
                print(f'epoch {epoch + 1}, loss: {train_l:.4f}')
            
    return est_W, train_l
```

---
## 3. `Softmax`回归模型的参数学习
- 方法3: `torch.nn`
```python
class SofmaxRegresModel(torch.nn.Module): 
    def __init__(self, dim_in, dim_out):
        super(SofmaxRegresModel, self).__init__() 
        self.layer1 = torch.nn.Linear(dim_in, dim_out, bias=True)
        
    def forward(self, x):
        y_pred = self.layer1(x)
        return torch.nn.functional.softmax(y_pred, dim=1)  # softmax

dim_in = X.shape[1]
dim_out = y.shape[1]
# 实例化1个网络
net = SofmaxRegresModel(dim_in, dim_out)
# 初始化网络参数和偏置
net.layer1.weight.data = torch.randn(dim_out, dim_in)
net.layer1.bias.data = torch.Tensor(dim_out)
# 损失函数
loss = torch.nn.CrossEntropyLoss()
# 随机梯度下降算法
trainer = torch.optim.SGD(net.parameters(), lr=0.05)
```

---
## 3. `Softmax`回归模型的参数学习
- 方法3: `torch.nn`

```python
# 加载数据
batch_size = 20
num_epochs = 100
dataset = TensorDataset(train_X, train_indices_y)
data_iter = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
# 开始训练
for epoch in range(num_epochs):
    for t_x, t_y in data_iter:
        l = loss(net(t_x), t_y)  # 计算当前批量的交叉熵损失
        trainer.zero_grad()  # 参数梯度清零
        l.backward()  # 反向传播，计算梯度
        trainer.step()  # 更新参数
    if (epoch+1) % 20 == 0:
        with torch.no_grad():  # 不计算梯度，加速损失函数的运算
            l_epoch = loss(net(train_X), train_indices_y) 
            print('epoch {}, loss {}'.format(epoch+1, l_epoch))

```
