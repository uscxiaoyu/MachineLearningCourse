import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import numpy as np


def grad_desc(lossfunc, w, x_dict, learn_rate=0.05, max_iters=1000):
    """
    f: 待优化目标函数, grad_f: f的梯度, w: 参数初值, x_dict:固定参数值, learn_rate: 学习率
    """
    trace_w = w.clone().data.reshape(1, -1)
    i = 0
    for i in range(max_iters):
        l = lossfunc(w, **x_dict)
        l.backward()
        w.data.sub_(learn_rate * w.grad.data)
        with torch.no_grad():
            trace_w = torch.cat([trace_w, w.detach().data.reshape(1, -1)], 0)
            if (i + 1) % 10 == 0:
                loss = lossfunc(w, **x_dict).data.numpy()
                print(f"迭代次数: {i+1}, 损失函数值: {loss:.4f}")

            if torch.sum(torch.abs(trace_w[-1] - trace_w[-2])) < 1e-3:  # 停止条件
                break

        w.grad.zero_()

    print(f"共迭代{i}次, 损失函数值: {lossfunc(w, **x_dict).data.numpy():.4f}, 最优参数值: {w.tolist()}")
    return trace_w


def grad_desc_with_momentum(lossfunc, w, x_dict, beta=0.5, learn_rate=0.05, max_iter=1000):
    trace_w = w.clone().data.reshape(1, -1)
    v_0 = 0
    i = 1
    while i <= max_iter:
        l = lossfunc(w, **x_dict)
        l.backward()
        v_1 = beta * v_0 + learn_rate * w.grad.data
        w.data.sub_(v_1)
        with torch.no_grad():
            trace_w = torch.cat([trace_w, w.detach().data.reshape(1, -1)], 0)
            if i % 10 == 0:
                loss = lossfunc(w, **x_dict).data.numpy()
                print(f"迭代次数: {i}, 损失函数值: {loss:.4f}")

            if torch.sum(torch.abs(trace_w[-1] - trace_w[-2])) < 1e-3:  # 停止条件
                break

        w.grad.zero_()
        v_0 = v_1
        i += 1

    print(f"共迭代{i-1}次, 损失函数值: {lossfunc(w, **x_dict).data.numpy():.4f}, 最优参数值: {w.tolist()}")
    return trace_w


def adaptive_momentum(lossfunc, w, x_dict, beta1=0.5, beta2=0.9, learn_rate=0.999, max_iter=1000, epsilon=1e-8):
    trace_w = w.clone().data.reshape(1, -1)
    v_0, s_0 = 0, 0
    i = 1
    while i <= max_iter:
        l = lossfunc(w, **x_dict)
        l.backward()
        v_1 = (beta1 * v_0 + (1 - beta1) * w.grad.data) / (1 - beta1 ** i)
        s_1 = (beta2 * s_0 + (1 - beta2) * w.grad.data ** 2) / (1 - beta2 ** i)
        w.data.sub_(learn_rate * v_1 / (torch.sqrt(s_1) + epsilon))
        with torch.no_grad():
            trace_w = torch.cat([trace_w, w.detach().data.reshape(1, -1)], 0)
            if i % 10 == 0:
                loss = lossfunc(w, **x_dict).data.numpy()
                print(f"迭代次数: {i}, 损失函数值: {loss:.4f}")

            if torch.sum(torch.abs(trace_w[-1] - trace_w[-2])) < 1e-3:  # 停止条件
                break

        w.grad.zero_()
        v_0, s_0 = v_1, s_1
        i += 1

    print(f"共迭代{i - 1}次, 损失函数值: {lossfunc(w, **x_dict).data.numpy():.4f}, 最优参数值: {w.tolist()}")
    return trace_w


if __name__ == "__main__":

    # res = grad_desc(f, grad_f, x0=np.array([3, 3]), learn_rate=0.2)
    res = grad_desc_with_momentum(f, grad_f, x0=np.array([3, 3]), beta=0.5, learn_rate=0.2)
    # res = adaptive_momentum(f, grad_f, x0=np.array([3, 3]), beta1=0.5, beta2=0.5, learn_rate=0.2)

    # 绘制动画
    a0, a1 = res[:, 0].tolist(), res[:, 1].tolist()
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    (line,) = ax.plot([], [], "-o", lw=0.5, color="orange")
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
        return (line,)

    def run(data):
        u, v = data
        print(u, v)
        xdata.append(u)
        ydata.append(v)
        line.set_data(xdata, ydata)
        return (line,)

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
