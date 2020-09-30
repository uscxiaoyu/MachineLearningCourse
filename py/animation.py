import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def data_gen():
    cnt = 0
    while cnt < 1000:
        t = cnt / 10
        yield t, np.sin(2 * np.pi * t) * np.exp(-t / 10.0)
        cnt += 1


def init():
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlim(0, 10)
    xdata, ydata = [], []
    line.set_data(xdata, ydata)
    return (line,)


fig, ax = plt.subplots()
(line,) = ax.plot([], [], lw=2)
ax.grid()
xdata, ydata = [], []


def run(data):
    # 更新绘图数据, 接收date_gen的返回值
    t, y = data
    xdata.append(t)
    ydata.append(y)
    xmin, xmax = ax.get_xlim()

    if t >= xmax:
        ax.set_xlim(xmin, 2 * xmax)
        ax.figure.canvas.draw()
    line.set_data(xdata, ydata)

    return (line,)  # 返回需绘制的内容


ani = animation.FuncAnimation(fig, run, data_gen, interval=10, init_func=init)
plt.show()
