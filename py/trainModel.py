import numpy as np
import torch
import matplotlib.pyplot as plt


def learn_params(lossfunc, X, y, epochs=50, lr=0.03):
    '''
    lossfunc: 损失函数
    X: 特征矩阵
    y: 标签
    epochs: 训练批次
    lr: 学习率
    '''
    params = torch.randn(size=(X.shape[1] + 1, 1), requires_grad=True)
    for epoch in range(epochs):
        loss = lossfunc(params, X, y)
        loss.backword()
        params.data.sub_(lr * params.grad)
        params.grad.data.zero_()
        with torch.no_grad():  # 不计算梯度，加速损失函数的运算
            d_params = params.detach()  # 从计算图中解绑，后面的操作不影响计算图中对应的结果
            train_loss = lossfunc(d_params, X, y)  # 最近一次的负对数似然率
            if epoch % 5 == 0:
                print(f'epoch {epoch}, loss: {train_loss.numpy():.4f}')
    return params


if __name__ == "__main__":
    def loss(w, X, y):  # 所有误分类点的误差
        '''
        w: 参数向量 n, 1
        X: 矩阵 m, n
        y: 向量 m, 1
        b: 偏置, 标量
        '''
        w_X = torch.cat([X, torch.ones((X.shape[0], 1))], axis=1)  # 增广
        hat_y = w_X @ w
        neg_Dist = -hat_y.reshape(1, -1) * y.reshape(1, -1)  # 误分类点对应的值为正
        return torch.sum(torch.relu(neg_Dist))  # relu取所有正值, 负值重置为0

    x0 = torch.randn(100, 2) + 2  # 均值为 2
    y0 = torch.ones(100)
    x1 = torch.randn(100, 2) - 2  # 均值为 -3
    y1 = -torch.ones(100)

    x = torch.cat((x0, x1)).type(torch.FloatTensor)
    y = torch.cat((y0, y1)).type(torch.LongTensor)

    idx = np.arange(len(x))
    np.random.shuffle(idx)  # 随机打乱索引次序
    train_x, train_y = x[idx[:50]], y[idx[:50]]  # 随机选取50个
    test_x, test_y = x[idx[50:]], y[idx[50:]]

    x_list = [train_x, test_x]
    y_list = [train_y, test_y]

    params = learn_params(loss, train_x, train_y, epochs=20)

    d_params = params.data.numpy()
    w0, w1, b = d_params.reshape(1, -1)[0]
    title_list = ['Train', 'Test']

    fig = plt.figure(figsize=(12, 5))
    for i in range(2):
        px = x_list[i]
        py = y_list[i]
        x0 = px.data.numpy()[:, 0]
        x1 = px.data.numpy()[:, 1]
        y = -(w0*x0+b) / w1  # 超平面
        ax = fig.add_subplot(1, 2, i+1)
        ax.set_title(title_list[i])
        ax.set_xlabel('$x_0$', fontsize=14)
        ax.set_ylabel('$x_1$', fontsize=14)
        ax.scatter(x0, x1, c=py.data.numpy(), s=20, lw=0, cmap='RdYlGn')
        ax.set_ylim([-5, 5])
        ax.plot(x0, y, 'k-', alpha=0.5,
                label=f"${w0:.2f}x_0+{w1:.2f}x_1+{b:0.2f}=0$")
        ax.legend(loc='best')

    plt.show()
