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

    def draw_derivative(self, points):
        folder_name = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        figure_save_path = os.path.join('./data/exp/', 'derivative_' + folder_name)
        os.mkdir(figure_save_path)
        for i in range(len(points)):
            derivative = self.derivative(points[i])
            # derivative = derivative/4
            plt.figure(figsize=(15, 15))
            x = np.linspace(-1.5, 1.5, 300)
            y = np.linspace(-1.5, 1.5, 300)
            X, Y = np.meshgrid(x, y)  # 获得网格坐标矩阵
            c = plt.contour(X, Y, self.__call__(np.array([X, Y])), 40, colors='green')
            plt.clabel(c, inline=True, fontsize=10)
            plt.quiver(points[i][0], points[i][1], points[i][0] + derivative[0], points[i][1] + derivative[1])

            plt.xticks(())
            plt.yticks(())
            plt.savefig(os.path.join(figure_save_path, str(i) + '.jpg'))

    def draw_counter(self, point_sequence=None):
        figure_save_path = None
        if point_sequence is not None:
            folder_name = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            figure_save_path = os.path.join('./data/exp/', 'frame_' + folder_name)
            os.mkdir(figure_save_path)
        if point_sequence is None:
            point_sequence = []
        x = np.linspace(-2.5, 2.5, 500)
        y = np.linspace(-2.5, 2.5, 500)
        X, Y = np.meshgrid(x, y)  # 获得网格坐标矩阵
        # fill color
        # plt.contourf(X, Y, self.__call__(X, Y), 8, cmap=plt.cm.hot)
        # draw contour
        plt.figure(figsize=(15, 15))
        c = plt.contour(X, Y, self.__call__(np.array([X, Y])), 80, colors='green')
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
        return (np.sin(4 * y) + x ** 2 + y ** 2) * np.exp(0.1 * (x + y))

    def derivative(self, point):
        x, y = point
        g = (np.sin(4 * y) + x ** 2 + y ** 2)
        h = np.exp(0.1 * (x + y))
        g_x = 2 * x
        g_y = np.cos(4 * y) * 4 + 2 * y
        h_x = h * (0.1 * (x + y)) * 0.1
        h_y = h * (0.1 * (x + y)) * 0.1
        x_ = h * g_x + g * h_x
        y_ = h * g_y + g * h_y
        return np.array([x_, y_])

    def hessian(self, point):
        x, y = point
        g = (np.sin(4 * y) + x ** 2 + y ** 2)
        h = np.exp(0.1 * (x + y))
        g_x = 2 * x
        g_xx = 2
        g_xy = 0
        g_y = np.cos(4 * y) * 4 + 2 * y
        g_yx = 0
        g_yy = -np.sin(4 * y) * 4 * 4 + 2
        h_y = h_x = h * (0.1 * (x + y)) * 0.1
        h_yx = h_xx = h_x * (0.1 * (x + y)) * 0.1 + 0.01 * h
        h_yy = h_xy = h_y * (0.1 * (x + y)) * 0.1 + 0.01 * h
        # h_y = h * (0.1 * (x + y)) * 0.1
        # h_yx = h_x*(0.1 * (x + y)) * 0.1 + 0.01*h
        # h_yy = h_y*(0.1 * (x + y)) * 0.1 + 0.01*h
        xx = h_x * g_x + h * g_xx + g_x * h_x + g * h_xx
        xy = h_y * g_x + h * g_xy + g_y * h_x + g * h_xy
        yx = h_x * g_y + h * g_yx + g_x * h_y + g * h_yx
        yy = h_y * g_y + h * g_yy + g_y * h_y + g * h_yy
        return np.array([[xx, xy], [yx, yy]])


class Quadratic(ObjectiveFunction):
    def __init__(self):
        super(Quadratic, self).__init__()

    def __call__(self, point):
        x, y = point
        return 2 * x ** 2 + y ** 2

    def derivative(self, point):
        x, y = point
        x_ = 4 * x
        y_ = 2 * y
        return np.array([x_, y_])


if __name__ == '__main__':
    of = OFExample()
    # generate a sequence with 10 point
    sequence_ = []
    for i in range(10):
        sequence_.append(np.array([random.random() * 2 - 1., random.random() * 2 - 1.]))
    # of.draw_counter(sequence_)
    of.draw_derivative(sequence_)
