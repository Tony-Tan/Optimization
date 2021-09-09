import numpy as np
import matplotlib.pyplot as plt
import os
import time
from rich.progress import track
import random

# if you work with pycharm, to make rich package work
# you have to enable “emulate terminal” in output console
# option in run/debug configuration to see styled output.
# https://rich.readthedocs.io/en/stable/introduction.html


class ObjectiveFunction:
    def __init__(self):
        pass

    def __call__(self, point):
        pass

    def derivative(self, point):
        pass

    def draw_counter(self, point_sequence=None):
        figure_save_path = None
        if point_sequence is not None:
            folder_name = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            figure_save_path = os.path.join('./data/exp/', folder_name)
            os.mkdir(figure_save_path)
        if point_sequence is None:
            point_sequence = []
        x = np.linspace(-1.5, 1.5, 300)
        y = np.linspace(-1.5, 1.5, 300)
        X, Y = np.meshgrid(x, y)  # 获得网格坐标矩阵
        # fill color
        # plt.contourf(X, Y, self.__call__(X, Y), 8, cmap=plt.cm.hot)
        # draw contour
        plt.figure(figsize=(15, 15))
        c = plt.contour(X, Y, self.__call__(np.array([X, Y])), 40, colors='green')
        plt.clabel(c, inline=True, fontsize=10)
        for i in track(range(len(point_sequence)), description="Generating Frame and saving in " + figure_save_path):
            if point_sequence is not None:
                if i > 0:
                    plt.plot([point_sequence[i][0], point_sequence[i - 1][0]],
                             [point_sequence[i][1], point_sequence[i - 1][1]],
                             c='r', alpha=0.8, linestyle='-.')
                plt.scatter(point_sequence[i][0], point_sequence[i][1], s=100,
                            c='b', alpha=0.8, marker='x')
                plt.xticks(())
                plt.yticks(())
                plt.savefig(os.path.join(figure_save_path, str(i) + '.jpg'))


class OFExample(ObjectiveFunction):
    def __init__(self):
        super(OFExample, self).__init__()

    def __call__(self, point):
        x, y = point
        return -(1 - x / 2 + x ** 5 + 0.3 * np.sin(5 * y ** 3)) * np.exp(-x ** 2 - y ** 2)

    def derivative(self, point):
        x, y = point
        g = -(1. - x / 2. + x ** 5 + 0.3 * np.sin(5 * y ** 3))
        g_x = -(-1 / 2 + 5 * x ** 4)
        g_y = -(0.3 * np.cos(5 * y ** 3) * 15 * y * 2)
        h = -x ** 2 - y ** 2
        h_x = -2 * x
        h_y = -2 * y
        x_ = g_x * np.exp(h) + g * h * np.exp(h) * h_x
        y_ = g_y * np.exp(h) + g * h * np.exp(h) * h_y
        return np.array([x_, y_])


class Quadratic(ObjectiveFunction):
    def __init__(self):
        super(Quadratic, self).__init__()

    def __call__(self, point):
        x, y = point
        return 2*x**2+y**2

    def derivative(self, point):
        x, y = point
        x_ = 4*x
        y_ = 2*y
        return np.array([x_, y_])


if __name__ == '__main__':
    of = Quadratic()
    # generate a sequence with 10 point
    sequence_ = []
    for i in range(10):
        sequence_.append(np.array([random.random() * 2 - 1., random.random() * 2 - 1.]))
    of.draw_counter(sequence_, True)

