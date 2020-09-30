import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import torch


def grad_desc(f, grad_f, x0, learn_rate=0.05):
    """
    f: 待优化目标函数, grad_f: f的梯度, x0: 参数初值, learn_rate: 学习率
    """
    trace_x = np.array([x0])  # x的历史记录
    x = x0
    i = 0
    while True:
        x = x - learn_rate * grad_f(x)  # 更新x的值
        trace_x = np.concatenate([trace_x, x.reshape(1, -1)])
        i += 1
        if i % 5 == 0:
            print(f"迭代次数: {i}, 目标函数值f: {f(x):.6f}")

        if np.sum(np.abs(trace_x[-1] - trace_x[-2])) < 1e-3:  # 停止条件
            break

    print(f"共迭代{len(trace_x)}次, 目标函数: {f(x)}, 最优参数值: {x.tolist()}")
    return trace_x


def grad_desc_with_momentum(f, grad_f, x0, beta=0.5, learn_rate=0.05):
    trace_x = np.array([x0])
    x = x0
    m_0 = 0
    i = 0
    while True:
        grad = grad_f(x)
        m_1 = beta * m_0 + (1 - beta) * grad
        x = x - learn_rate * m_1
        trace_x = np.concatenate([trace_x, x.reshape(1, -1)])
        if i % 5 == 0:
            print(f"迭代次数: {i}, 目标函数值f: {f(x):.6f}")

        if np.sum(np.abs(trace_x[-1] - trace_x[-2])) < 1e-3:  # 停止条件
            break

        m_0 = m_1
        i += 1

    print(f"共迭代{len(trace_x)}次, 目标函数: {f(x)}, 最优参数值: {x.tolist()}")
    return trace_x


def adaptive_momentum(f, grad_f, x0, beta1=0.5, beta2=0.5, learn_rate=0.05):
    trace_x = np.array([x0])
    x = x0
    m_0, v_0 = 0, 0
    i = 0
    while True:
        grad = grad_f(x)
        m_1 = beta1 * m_0 + (1 - beta1) * grad
        v_1 = beta2 * v_0 + (1 - beta2) * grad ** 2
        x = x - learn_rate * m_1 / np.sqrt(v_1)
        trace_x = np.concatenate([trace_x, x.reshape(1, -1)])
        if i % 5 == 0:
            print(f"迭代次数: {i}, 目标函数值f: {f(x):.6f}")

        if np.sum(np.abs(trace_x[-1] - trace_x[-2])) < 1e-3:  # 停止条件
            break

        m_0, v_0 = m_1, v_1
        i += 1

    print(f"共迭代{len(trace_x)}次, 目标函数: {f(x)}, 最优参数值: {x.tolist()}")
    return trace_x


if __name__ == "__main__":

    def f(x):
        """
        函数: f(x0, x1) = x0**2 + x1**2
        """
        return x[0] ** 2 + 2 * x[1] ** 2  # objective

    def grad_f(x):
        """
        f(x)的梯度
        """
        return np.array([2 * x[0], 4 * x[1]])  # gradient

    # res = grad_desc(f, grad_f, x0=np.array([3, 3]), learn_rate=0.2)
    res = grad_desc_with_momentum(f, grad_f, x0=np.array([3, 3]), beta=0.5, learn_rate=0.2)
    # res = adaptive_momentum(f, grad_f, x0=np.array([3, 3]), beta1=0.5, beta2=0.5, learn_rate=0.2)

    # 绘制动画
    a0, a1 = res[:, 0].tolist(), res[:, 1].tolist()
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    line, = ax.plot([], [], "-o", lw=0.5, color="orange")
    ax.grid(False)
    xdata, ydata = [], []
    x0 = np.arange(-5.5, 5.0, 0.1)
    x1 = np.arange(min(-3.0, min(a0) - 1), max(1.0, max(a1) + 1), 0.1)
    x0, x1 = np.meshgrid(x0, x1)
    ax.contour(x0, x1, f([x0, x1]), colors="grey", linewidths=1, alpha=0.2, linestyles="solid")
    ax.set_xlabel("$x_0$")
    ax.set_ylabel("$x_1$")

    def data_gen():
        for u, v in zip(a0, a1):
            yield u, v

    def init():
        xdata, ydata = [], []
        line.set_data(xdata, ydata)
        return line,

    def run(data):
        u, v = data
        print(u, v)
        xdata.append(u)
        ydata.append(v)
        line.set_data(xdata, ydata)
        return line,

    ani = animation.FuncAnimation(fig, run, data_gen, interval=1000, init_func=init, repeat=False)
    plt.show()

    # 静图
    # plt.figure(figsize=(10, 6))
    # plt.plot(x0, x1, "-o", color="#ff7f0e")
    # x0 = np.arange(-5.5, 5.0, 0.1)
    # x1 = np.arange(min(-3.0, min(x1) - 1), max(1.0, max(x1) + 1), 0.1)
    # x0, x1 = np.meshgrid(x0, x1)
    # plt.contour(x0, x1, f([x0, x1]), colors="#1f77b4", linewidths=1, linestyles="dashed")
    # plt.xlabel("x0")
    # plt.ylabel("x1")
    # plt.show()

