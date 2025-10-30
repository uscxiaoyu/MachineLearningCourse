import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 创建x轴数据
x = np.linspace(-6, 6, 400)

# 定义Sigmoid函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 定义Tanh函数
def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

# 计算函数值
y_sigmoid = sigmoid(x)
y_tanh = tanh(x)

# 创建图形和子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 绘制Sigmoid函数
ax1.plot(x, y_sigmoid, 'b-', linewidth=2, label='Sigmoid')
ax1.set_title('Sigmoid 函数', fontsize=14, fontweight='bold')
ax1.set_xlabel('x')
ax1.set_ylabel('f(x)')
ax1.grid(True, alpha=0.3)
ax1.legend()
# ax1.vlines(0)
ax1.set_ylim(-0.1, 1.1)

# 添加Sigmoid特性标注
ax1.text(-5.5, 0.8, '输出范围: (0, 1)', fontsize=10)
ax1.text(-5.5, 0.7, '用于二分类问题', fontsize=10)
ax1.text(-5.5, 0.6, '梯度消失问题', fontsize=10, color='red')

# 绘制Tanh函数
ax2.plot(x, y_tanh, 'r-', linewidth=2, label='Tanh')
ax2.set_title('Tanh 函数', fontsize=14, fontweight='bold')
ax2.set_xlabel('x')
ax2.set_ylabel('f(x)')
# ax2.vlines(0)
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.set_ylim(-1.1, 1.1)

# 添加Tanh特性标注
ax2.text(-5.5, 0.8, '输出范围: (-1, 1)', fontsize=10)
ax2.text(-5.5, 0.6, '均值为0，训练更稳定', fontsize=10)

# 添加水平参考线
ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax1.axhline(y=1, color='k', linestyle='--', alpha=0.3)
ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax2.axhline(y=1, color='k', linestyle='--', alpha=0.3)
ax2.axhline(y=-1, color='k', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()
