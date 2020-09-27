import matplotlib.pyplot as plt
import numpy as np


def grad_desc(f, grad_f, x0, learn_rate=0.05):
    '''
    f: 待优化目标函数, grad_f: f的梯度, x0: 参数初值, learn_rate: 学习率
    '''
    trace_x = np.array([x0])  # x的历史记录
    x = x0
    i = 0
    while True:
        x = x - learn_rate * grad_f(x)  # 更新x的值
        trace_x = np.concatenate([trace_x, x.reshape(1, -1)])
        i += 1
        if i % 5 == 0:
            print(f'迭代次数: {i}, 目标函数值f: {f(x):.6f}')

        if np.sum(np.abs(x)) < 1e-3:  # 停止条件
            break

    print(f'共迭代{len(trace_x)}次, 目标函数: {f(x)}, 最优参数值: {x.tolist()}')
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
            print(f'迭代次数: {i}, 目标函数值f: {f(x):.6f}')

        if np.sum(np.abs(x)) < 1e-3:  # 停止条件
            break

        m_0 = m_1
        i += 1

    print(f'共迭代{len(trace_x)}次, 目标函数: {f(x)}, 最优参数值: {x.tolist()}')
    return trace_x


def adaptive_momentum(f, grad_f, x0, beta1=0.5, beta2=0.5, learn_rate=0.05):
    trace_x = np.array([x0])
    x = x0
    m_0, v_0 = 0, 0
    i = 0
    while True:
        grad = grad_f(x)
        m_1 = beta1 * m_0 + (1 - beta1) * grad
        v_1 = beta2 * v_0 + (1 - beta2) * grad**2
        x = x - learn_rate * m_1 / np.sqrt(v_1)
        trace_x = np.concatenate([trace_x, x.reshape(1, -1)])
        if i % 5 == 0:
            print(f'迭代次数: {i}, 目标函数值f: {f(x):.6f}')

        if np.sum(np.abs(x)) < 1e-3:  # 停止条件
            break

        m_0, v_0 = m_1, v_1
        i += 1

    print(f'共迭代{len(trace_x)}次, 目标函数: {f(x)}, 最优参数值: {x.tolist()}')
    return trace_x


if __name__ == "__main__":
    def f(x):
        '''
        函数: f(x0, x1) = x0**2 + x1**2
        '''
        return x[0]**2 + 2 * x[1]**2   # objective

    def grad_f(x):
        '''
        f(x)的梯度
        '''
        return np.array([2 * x[0], 4 * x[1]])    # gradient

    # res = grad_desc(f, grad_f, x0=np.array([3, 3]), learn_rate=0.2)
    # res = grad_desc_with_momentum(f, grad_f, x0=np.array([3, 3]), beta=0.5, learn_rate=0.2)
    res = adaptive_momentum(f, grad_f, x0=np.array([3, 3]), beta1=0.5, beta2=0.5, learn_rate=0.2)
    x0, x1 = res[:, 0], res[:, 1]
    plt.figure(figsize=(10, 6))
    plt.plot(x0, x1, '-o', color='#ff7f0e')
    x0 = np.arange(-5.5, 5.0, 0.1)
    x1 = np.arange(min(-3.0, min(x1) - 1), max(1.0, max(x1) + 1), 0.1)
    x0, x1 = np.meshgrid(x0, x1)
    plt.contour(x0, x1, f([x0, x1]), colors='#1f77b4')
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.show()
