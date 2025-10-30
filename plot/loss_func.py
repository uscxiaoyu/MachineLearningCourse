import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# 设置中文字体（根据系统选择合适的字体）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']  
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ========== 1. 定义目标函数（损失函数）==========
def loss_function(x):
    """
    定义一个非凸的损失函数：
    L(x) = 0.1 * x^4 - 1.5 * x^3 + 6 * x^2 - 5 * x + 10
    这个函数有多个局部最优点，更贴近实际神经网络的损失曲面
    """
    return 0.1 * x**4 - 1.5 * x**3 + 6 * x**2 - 5 * x + 10

# ========== 2. 定义目标函数的梯度（导数）==========
def gradient(x):
    """
    损失函数的导数（梯度）：
    dL/dx = 0.4 * x^3 - 4.5 * x^2 + 12 * x - 5
    梯度指向函数增长最快的方向，我们要沿着负梯度方向下降
    """
    return 0.4 * x**3 - 4.5 * x**2 + 12 * x - 5

# ========== 3. 梯度下降算法实现 ==========
def gradient_descent(start_x, learning_rate=0.01, num_iterations=100):
    """
    梯度下降优化算法
    
    参数：
    - start_x: 初始参数位置（起点）
    - learning_rate: 学习率，控制每次更新的步长
    - num_iterations: 迭代次数
    
    返回：
    - x_history: 参数更新的历史轨迹
    - loss_history: 每次迭代的损失值
    """
    x = start_x
    x_history = [x]
    loss_history = [loss_function(x)]
    
    for i in range(num_iterations):
        # 计算当前位置的梯度
        grad = gradient(x)
        
        # 梯度下降更新规则：x_new = x_old - learning_rate * gradient
        x = x - learning_rate * grad
        
        # 记录轨迹
        x_history.append(x)
        loss_history.append(loss_function(x))
    
    return np.array(x_history), np.array(loss_history)

# ========== 4. 生成数据 ==========
# 生成损失函数曲线的 x 坐标
x_range = np.linspace(-2, 8, 500)
y_range = loss_function(x_range)

# 运行梯度下降算法（从不同的起点）
start_point_1 = 7.0  # 起点1：靠右的位置
start_point_2 = 0.5  # 起点2：靠左的位置

x_path_1, loss_path_1 = gradient_descent(start_point_1, learning_rate=0.02, num_iterations=80)
x_path_2, loss_path_2 = gradient_descent(start_point_2, learning_rate=0.02, num_iterations=80)

# ========== 5. 绘制静态图 ==========
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 左图：损失函数曲面与优化路径
ax1.plot(x_range, y_range, 'b-', linewidth=2, label='损失函数 L(θ)')
ax1.plot(x_path_1, loss_path_1, 'ro-', markersize=4, linewidth=1.5, alpha=0.7, label='优化路径1（起点=7.0）')
ax1.plot(x_path_2, loss_path_2, 'go-', markersize=4, linewidth=1.5, alpha=0.7, label='优化路径2（起点=0.5）')
ax1.scatter([x_path_1[0]], [loss_path_1[0]], color='red', s=100, zorder=5, label='起点1')
ax1.scatter([x_path_2[0]], [loss_path_2[0]], color='green', s=100, zorder=5, label='起点2')
ax1.scatter([x_path_1[-1]], [loss_path_1[-1]], color='darkred', s=150, marker='*', zorder=5, label='终点1')
ax1.scatter([x_path_2[-1]], [loss_path_2[-1]], color='darkgreen', s=150, marker='*', zorder=5, label='终点2')

ax1.set_xlabel('参数 θ', fontsize=12)
ax1.set_ylabel('损失值 L(θ)', fontsize=12)
ax1.set_title('梯度下降优化过程 - 损失曲面视角', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# 右图：损失值随迭代次数的变化
ax2.plot(range(len(loss_path_1)), loss_path_1, 'r-', linewidth=2, label='路径1损失下降')
ax2.plot(range(len(loss_path_2)), loss_path_2, 'g-', linewidth=2, label='路径2损失下降')
ax2.set_xlabel('迭代次数 (Iteration)', fontsize=12)
ax2.set_ylabel('损失值 L(θ)', fontsize=12)
ax2.set_title('损失值的收敛过程', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('optimization_process.png', dpi=300, bbox_inches='tight')
plt.show()

# ========== 6. 打印关键信息 ==========
print("=" * 60)
print("梯度下降优化结果")
print("=" * 60)
print(f"路径1 - 起点: θ₀ = {start_point_1:.2f}, 损失: L = {loss_path_1[0]:.2f}")
print(f"路径1 - 终点: θ* = {x_path_1[-1]:.2f}, 损失: L = {loss_path_1[-1]:.2f}")
print(f"路径1 - 迭代次数: {len(x_path_1) - 1}, 损失下降: {loss_path_1[0] - loss_path_1[-1]:.2f}")
print("-" * 60)
print(f"路径2 - 起点: θ₀ = {start_point_2:.2f}, 损失: L = {loss_path_2[0]:.2f}")
print(f"路径2 - 终点: θ* = {x_path_2[-1]:.2f}, 损失: L = {loss_path_2[-1]:.2f}")
print(f"路径2 - 迭代次数: {len(x_path_2) - 1}, 损失下降: {loss_path_2[0] - loss_path_2[-1]:.2f}")
print("=" * 60)
print("\n💡 关键观察：")
print("1. 不同起点可能收敛到不同的局部最优点")
print("2. 学习率太大会导致震荡，太小会导致收敛缓慢")
print("3. 梯度（斜率）越大，参数更新的步长越大")
print("=" * 60)
