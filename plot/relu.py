import numpy as np
import matplotlib.pyplot as plt
from matplotlib import mathtext

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
plt.rcParams['mathtext.default'] = 'regular'  # 设置数学文本格式
plt.rcParams['mathtext.fontset'] = 'stix'     # 使用更兼容的数学字体集

# 创建x轴数据
x = np.linspace(-3, 3, 400)

# 定义ReLU函数
def relu(x):
    return np.maximum(0, x)

# 定义Leaky ReLU函数
def leaky_relu(x, alpha=0.1):
    return np.where(x > 0, x, alpha * x)

# 定义ELU函数
def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

# 计算函数值
y_relu = relu(x)
y_leaky_relu = leaky_relu(x)
y_elu = elu(x)

# 创建图形和子图
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 绘制ReLU函数
axes[0].plot(x, y_relu, 'b-', linewidth=2.5, label='ReLU')
axes[0].set_title('ReLU 函数', fontsize=14, fontweight='bold')
axes[0].set_xlabel('x')
axes[0].set_ylabel('f(x)')
# axes[0].grid(True, alpha=0.3)  # 去掉网格线
axes[0].legend()
axes[0].set_ylim(-1.5, 3.5)

# 添加ReLU特性标注
axes[0].text(-2.8, 3.0, r'公式: $f(x) = \max(0, x)$', fontsize=10)
axes[0].text(-2.8, 2.6, '输出范围: [0, +∞)', fontsize=10)
axes[0].text(-2.8, 2.2, '优点: 计算简单，无饱和区', fontsize=10, color='green')
axes[0].text(-2.8, 1.8, '缺点: 神经元"死亡"问题', fontsize=10, color='red')

# 绘制Leaky ReLU函数
axes[1].plot(x, y_leaky_relu, 'r-', linewidth=2.5, label='Leaky ReLU (α=0.1)')
axes[1].set_title('Leaky ReLU 函数', fontsize=14, fontweight='bold')
axes[1].set_xlabel('x')
axes[1].set_ylabel('f(x)')
# axes[1].grid(True, alpha=0.3)  # 去掉网格线
axes[1].legend()
axes[1].set_ylim(-1.5, 3.5)

# 添加Leaky ReLU特性标注
axes[1].text(-2.8, 3.0, r'公式: $f(x)=\alpha x\ (x\leq 0),\; f(x)=x\ (x>0)$', fontsize=10)
axes[1].text(-2.8, 2.6, '输出范围: (-∞, +∞)', fontsize=10)
axes[1].text(-2.8, 2.2, '优点: 解决神经元死亡问题', fontsize=10, color='green')
axes[1].text(-2.8, 1.8, '缺点: 需要选择α参数', fontsize=10, color='red')

# 绘制ELU函数
axes[2].plot(x, y_elu, 'g-', linewidth=2.5, label='ELU (α=1.0)')
axes[2].set_title('ELU 函数', fontsize=14, fontweight='bold')
axes[2].set_xlabel('x')
axes[2].set_ylabel('f(x)')
# axes[2].grid(True, alpha=0.3)  # 去掉网格线
axes[2].legend()
axes[2].set_ylim(-1.5, 3.5)

# 添加ELU特性标注
axes[2].text(-2.8, 3.0, r'公式: $f(x)=\alpha(e^x-1)\ (x\leq 0),\; f(x)=x\ (x>0)$', fontsize=10)
axes[2].text(-2.8, 2.6, '输出范围: (-α, +∞)', fontsize=10)
axes[2].text(-2.8, 2.2, '优点: 无死亡问题，均值接近0', fontsize=10, color='green')
axes[2].text(-2.8, 1.8, '缺点: 计算指数较复杂', fontsize=10, color='red')

# 在所有子图中添加参考线
for ax in axes:
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.5, linewidth=0.8)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.5, linewidth=0.8)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=0.5)
    ax.axhline(y=1, color='k', linestyle='--', alpha=0.3, linewidth=0.5)
    ax.axhline(y=-1, color='k', linestyle='--', alpha=0.3, linewidth=0.5)

plt.tight_layout()
plt.show()

# 可选：创建综合比较图
plt.figure(figsize=(12, 6))
plt.plot(x, y_relu, 'b-', linewidth=2, label='ReLU')
plt.plot(x, y_leaky_relu, 'r-', linewidth=2, label='Leaky ReLU (α=0.1)')
plt.plot(x, y_elu, 'g-', linewidth=2, label='ELU (α=1.0)')
plt.title('ReLU系列激活函数比较', fontsize=16, fontweight='bold')
plt.xlabel('x')
plt.ylabel('f(x)')
# plt.grid(True, alpha=0.3)  # 去掉网格线
plt.legend()
plt.axhline(y=0, color='k', linestyle='-', alpha=0.5)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.5)
plt.ylim(-1.5, 3.5)

# 添加关键点标注
key_points = [-2, -1, 0, 1, 2]
for point in key_points:
    plt.plot(point, relu(point), 'bo', markersize=4)
    plt.plot(point, leaky_relu(point), 'ro', markersize=4)
    plt.plot(point, elu(point), 'go', markersize=4)

plt.tight_layout()
plt.show()

# 打印函数在关键点的值
print("关键点函数值比较:")
print("x\tReLU\tLeaky ReLU\tELU")
for point in key_points:
    print(f"{point}\t{relu(point):.2f}\t{leaky_relu(point):.2f}\t\t{elu(point):.2f}")