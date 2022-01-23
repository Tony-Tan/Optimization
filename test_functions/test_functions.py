import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from rich.progress import track
import time
import os

PI = np.pi


class TestFunctions2D:
    def __init__(self, function_name, x_min, x_max, y_min, y_max, global_minimum):
        self.name = function_name
        self.search_domain = {'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max}
        self.search_range = {'x_range': x_max - x_min, 'y_range': y_max - y_min}
        self.global_minimum = global_minimum

    def __call__(self, *args):
        return 0

    @staticmethod
    def derivative(*args):
        return 0

    @staticmethod
    def Hessian_Matrix(*args):
        return 0

    def save(self, trajectory):
        folder_name = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        figure_save_path = os.path.join('./data/exp/', 'frame_' + folder_name)
        os.mkdir(figure_save_path)
        x = np.arange(self.search_domain['x_min'], self.search_domain['x_max'], self.search_range['x_range'] / 1000.)
        y = np.arange(self.search_domain['y_min'], self.search_domain['y_max'], self.search_range['y_range'] / 1000.)
        X, Y = np.meshgrid(x, y)
        Z = self.__call__([X, Y])
        plt.figure(figsize=(10, 10))
        CS = plt.contour(X, Y, Z, 40, alpha=0.7, cmap=mpl.cm.jet)
        for global_min_i in self.global_minimum:
            plt.scatter(global_min_i[0], global_min_i[1], c='r', marker='o')
        plt.clabel(CS, inline=True, fontsize=10)
        # plt.show()
        for i in track(range(len(trajectory)), description="Generating Frame and saving in " + figure_save_path):
            if trajectory is not None:
                point_i = trajectory[i]
                if i > 0:
                    plt.plot([point_i[0], point_i[0]],
                             [point_i[1], point_i[1]],
                             c='r', alpha=0.8, linestyle='-.')
                plt.scatter(point_i[0], point_i[1], s=100,
                            c='b', alpha=0.8, marker='x')
                plt.xticks(())
                plt.yticks(())
                plt.savefig(os.path.join(figure_save_path, str(i) + '.jpg'))


class AckleyFunction(TestFunctions2D):
    def __init__(self):
        super().__init__('Ackley Function', -5, 5, -5, 5, [[0, 0]])

    def __call__(self, x_k):
        x, y = x_k
        return -20 * np.exp(-0.2 * np.sqrt(0.5 * (x ** 2 + y ** 2))) - np.exp(
            0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) + np.e + 20

    @staticmethod
    def derivative(x_k):
        x, y = x_k
        part_1 = (2 * np.sqrt(2) * np.exp(-(np.sqrt(2 * (x ** 2 + y ** 2))) / 10)) / (np.sqrt(x ** 2 + y ** 2))
        part_2 = PI * np.exp(np.cos(2 * PI * x) / 2 + np.cos(2 * PI * y) / 2)
        partial_x = part_1 * x + part_2 * np.sin(2 * PI * x)
        partial_y = part_1 * y + part_2 * np.sin(2 * PI * y)
        return np.array([partial_x, partial_y])

    @staticmethod
    def Hessian_Matrix(x_k):
        pass


class SphereFunction(TestFunctions2D):
    def __init__(self):
        super().__init__('Sphere Function', -2, 2, -2, 2, [[0, 0]])

    def __call__(self, x_k):
        x, y = x_k
        return x ** 2 + y ** 2

    @staticmethod
    def derivative(x_k):
        x, y = x_k
        partial_x = 2 * x
        partial_y = 2 * y
        return np.array([partial_x, partial_y])

    @staticmethod
    def Hessian_Matrix(x_k):
        x, y = x_k
        partial_xx = 2
        partial_xy = 0
        partial_yx = 0
        partial_yy = 2
        return np.array([[partial_xx, partial_xy], [partial_yx, partial_yy]])


class RosenbrockFunction(TestFunctions2D):
    def __init__(self):
        super().__init__('Sphere Function', -2, 2, -1, 3, [[1, 1]])

    def __call__(self, x_k):
        x, y = x_k
        return 100 * (y - x ** 2) ** 2 + (1 - x) ** 2

    @staticmethod
    def derivative(x_k):
        x, y = x_k
        partial_x = -400 * x * y + 400 * x ** 3 + 2 * x - 2
        partial_y = -200 * x ** 2 + 200 * y
        return np.array([partial_x, partial_y])

    @staticmethod
    def Hessian_Matrix(x_k):
        x, y = x_k
        partial_xx = -400 * y + 1200 * x ** 2 + 2
        partial_xy = -400 * x
        partial_yx = -400 * x
        partial_yy = 200
        return np.array([[partial_xx, partial_xy], [partial_yx, partial_yy]])


class BealeFunction(TestFunctions2D):
    def __init__(self):
        super().__init__('Beale Function', -4.5, 4.5, -4.5, 4.5, [[3, 0.5]])

    def __call__(self, x_k):
        x, y = x_k
        return (1.5 - x + x * y) ** 2 + (2.25 - x + x * y ** 2) ** 2 + (2.625 - x + x * y ** 3) ** 2

    @staticmethod
    def derivative(x_k):
        x, y = x_k
        partial_x = 0.25 * (y - 1) * (
                8 * x * y ** 5 + 8 * x * y ** 4 + 16 * x * y ** 3 - 8 * x * y - 24 * x + 21 * y ** 2 + 39 * y + 51)
        partial_y = 0.25 * x * (
                24 * x * y ** 5 + 16 * x * y ** 3 - 24 * x * y ** 2 - 8 * x * y - 8 * x + 63 * y ** 2 + 36 * y + 12)
        return np.array([partial_x, partial_y])

    @staticmethod
    def Hessian_Matrix(x_k):
        x, y = x_k
        partial_xx = 0.25 * (y - 1) * (8 * y ** 5 + 8 * y ** 4 + 16 * y ** 3 - 8 * y - 24)
        partial_xy = 12 * x * y ** 5 + 8 * x * y ** 2 - 12 * x * y ** 2 - 4 * x + 63 * y ** 2 / 4 + 9 * y + 3
        partial_yx = 12 * x * y ** 5 + 8 * x * y ** 3 - 12 * x * y ** 2 - 4 * x + 63 * y ** 2 / 4 + 9 * y + 3
        partial_yy = x * (60 * x * y ** 4 + 24 * x * y ** 2 - 24 * x * y - 4 * x + 63 * y + 18) / 2.
        return np.array([[partial_xx, partial_xy], [partial_yx, partial_yy]])


class GoldsteinPriceFunction(TestFunctions2D):
    def __init__(self):
        super().__init__('Goldstein Price Function', -2, 2, -2, 2, [[0, -1]])

    def __call__(self, x_k):
        x, y = x_k
        return (1 + (x + y + 1) ** 2 * (19 - 14 * x + 3 * x ** 2 - 14 * y + 6 * x * y + 3 * y ** 2)) * (
                30 + (2 * x - 3 * y) ** 2 * (18 - 32 * x + 12 * x ** 2 + 48 * y - 36 * x * y + 27 * y ** 2))

    def derivative(self, x_k):
        x, y = x_k
        partial_x = 1152 * x ** 7 - 2016 * x ** 6 * y - 5376 * x ** 6 - 3888 * x ** 5 * y ** 2 + 8064 * x ** 5 * y + 5712 * x ** 5 + 6120 * x ** 4 * y ** 3 + \
                    12960 * x ** 4 * y ** 2 - 840 * x ** 4 * y + 6720 * x ** 4 + 5220 * x ** 3 * y ** 4 - 16320 * x ** 3 * y ** 3 - 21480 * x ** 3 * y ** 2 - \
                    30720 * x ** 3 * y - 9816 * x ** 3 - 5508 * x ** 2 * y ** 5 - 10440 * x ** 2 * y ** 4 + 3720 * x ** 2 * y ** 3 + 29520 * x ** 2 * y ** 2 + \
                    17352 * x ** 2 * y - 3216 * x ** 2 - 2916 * x * y ** 6 + 7344 * x * y ** 5 + 17460 * x * y ** 4 + 10080 * x * y ** 3 + 15552 * x * y ** 2 + \
                    14688 * x * y + 2520 * x + 972 * y ** 7 + 1944 * y ** 6 - 1188 * y ** 5 - 11880 * y ** 4 - 23616 * y ** 3 - 19296 * y ** 2 - 4680 * y + 720
        partial_y = - 288 * x ** 7 - 1296 * x ** 6 * y + 1344 * x ** 6 + 3672 * x ** 5 * y ** 2 + 5184 * x ** 5 * y - 168 * x ** 5 + 5220 * x ** 4 * y ** 3 - \
                    12240 * x ** 4 * y ** 2 - 10740 * x ** 4 * y - 7680 * x ** 4 - 9180 * x ** 3 * y ** 4 - 13920 * x ** 3 * y ** 3 + 3720 * x ** 3 * y ** 2 + 19680 * x ** 3 * y + \
                    5784 * x ** 3 - 8748 * x ** 2 * y ** 5 + 18360 * x ** 2 * y ** 4 + 34920 * x ** 2 * y ** 3 + 15120 * x ** 2 * y ** 2 + 15552 * x ** 2 * y + 7344 * x ** 2 + \
                    6804 * x * y ** 6 + 11664 * x * y ** 5 - 5940 * x * y ** 4 - 47520 * x * y ** 3 - 70848 * x * y ** 2 - 38592 * x * y - 4680 * x + 5832 * y ** 7 - \
                    45368 * y ** 6 - 26568 * y ** 5 + 9720 * y ** 4 + 57384 * y ** 3 + 36864 * y ** 2 + 6120 * y + 720
        return np.array([partial_x, partial_y])

    @staticmethod
    def Hessian_Matrix(x_k):
        x, y = x_k
        partial_xx = 8064 * x ** 6 - 12096 * x ** 5 * y - 32256 * x ** 5 - 19440 * x ** 4 * y ** 2 + 40320 * x ** 4 * y + 28560 * x ** 4 + 24480 * x ** 3 * y ** 3 + \
                     51840 * x ** 3 * y ** 2 - 3360 * x ** 3 * y + 26880 * x ** 3 + 15660 * x ** 2 * y ** 4 - 48960 * x ** 2 * y ** 3 - 64440 * x ** 2 * y ** 2 - \
                     92160 * x ** 2 * y - 29448 * x ** 2 - 11016 * x * y ** 5 - 20880 * x * y ** 4 + 7440 * x * y ** 3 + 59040 * x * y ** 2 + 34704 * x * y - \
                     6432 * x - 2916 * y ** 6 + 7344 * y ** 5 + 17460 * y ** 4 + 10080 * y ** 3 + 15552 * y ** 2 + 14688 * y + 2520

        partial_xy = - 2016 * x ** 6 - 7776 * x ** 5 * y + 8064 * x ** 5 + 18360 * x ** 4 * y ** 2 + 25920 * x ** 4 * y - \
                     840 * x ** 4 + 20880 * x ** 3 * y ** 3 - 48960 * x ** 3 * y ** 2 - 42960 * x ** 3 * y - \
                     30720 * x ** 3 - 27540 * x ** 2 * y ** 4 - 41760 * x ** 2 * y ** 3 + 11160 * x ** 2 * y ** 2 + \
                     59040 * x ** 2 * y + 17352 * x ** 2 - 17496 * x * y ** 5 + 36720 * x * y ** 4 + 69840 * x * y ** 3 + \
                     30240 * x * y ** 2 + 31104 * x * y + 14688 * x + 6804 * y ** 6 + 11664 * y ** 5 - 5940 * y ** 4 - \
                     47520 * y ** 3 - 70848 * y ** 2 - 38592 * y - 4680

        partial_yx = - 2016 * x ** 6 - 7776 * x ** 5 * y + 8064 * x ** 5 + 18360 * x ** 4 * y ** 2 + 25920 * x ** 4 * y - \
                     840 * x ** 4 + 20880 * x ** 3 * y ** 3 - 48960 * x ** 3 * y ** 2 - 42960 * x ** 3 * y - 30720 * x ** 3 - \
                     27540 * x ** 2 * y ** 4 - 41760 * x ** 2 * y ** 3 + 11160 * x ** 2 * y ** 2 + 59040 * x ** 2 * y + \
                     17352 * x ** 2 - 17496 * x * y ** 5 + 36720 * x * y ** 4 + 69840 * x * y ** 3 + 30240 * x * y ** 2 + \
                     31104 * x * y + 14688 * x + 6804 * y ** 6 + 11664 * y ** 5 - 5940 * y ** 4 - 47520 * y ** 3 - 70848 * y ** 2 - 38592 * y - 4680

        partial_yy = - 1296 * x ** 6 + 7344 * x ** 5 * y + 5184 * x ** 5 + 15660 * x ** 4 * y ** 2 - 24480 * x ** 4 * y - \
                     10740 * x ** 4 - 36720 * x ** 3 * y ** 3 - 41760 * x ** 3 * y ** 2 + 7440 * x ** 3 * y + 19680 * x ** 3 - \
                     43740 * x ** 2 * y ** 4 + 73440 * x ** 2 * y ** 3 + 104760 * x ** 2 * y ** 2 + 30240 * x ** 2 * y + \
                     15552 * x ** 2 + 40824 * x * y ** 5 + 58320 * x * y ** 4 - 23760 * x * y ** 3 - 142560 * x * y ** 2 - \
                     141696 * x * y - 38592 * x + 40824 * y ** 6 - 272208 * y ** 5 - 132840 * y ** 4 + 38880 * y ** 3 + 172152 * y ** 2 + 73728 * y + 6120

        return np.array([[partial_xx, partial_xy], [partial_yx, partial_yy]])


class BoothFunction(TestFunctions2D):
    def __init__(self):
        super().__init__('Booth Function', -10, 10, -10, 10, [[1, 3]])

    def __call__(self, x_k):
        x, y = x_k
        return (x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2

    def derivative(self, x_k):
        x, y = x_k
        partial_x = 10 * x + 8 * y - 34
        partial_y = 8 * x + 10 * y - 38
        return np.array([partial_x, partial_y])

    @staticmethod
    def Hessian_Matrix(x_k):
        x, y = x_k
        partial_xx = 10
        partial_xy = 8
        partial_yx = 8
        partial_yy = 10
        return np.array([[partial_xx, partial_xy], [partial_yx, partial_yy]])


class MatyasFunction(TestFunctions2D):
    def __init__(self):
        super(MatyasFunction, self).__init__('Matyas Function', -10, 10, -10, 10, [[0, 0]])

    def __call__(self, x_k):
        x, y = x_k
        return 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y

    def derivative(self, x_k):
        x, y = x_k
        partial_x = (13 * x - 12 * y) / 25
        partial_y = -12 * x / 25 + 13 * y / 25
        return np.array([partial_x, partial_y])

    @staticmethod
    def Hessian_Matrix(x_k):
        x, y = x_k
        partial_xx = 13 / 25
        partial_xy = - 12 / 25
        partial_yx = -12 / 25
        partial_yy = 13 / 25
        return np.array([[partial_xx, partial_xy], [partial_yx, partial_yy]])


class LeviFunctionN13(TestFunctions2D):
    def __init__(self):
        super(LeviFunctionN13, self).__init__('Levi Function N. 13', -10, 10, -10, 10, [[1, 1]])

    def __call__(self, x_k):
        x, y = x_k
        return np.sin(3 * np.pi * x) ** 2 + (x - 1) ** 2 * (1 + np.sin(3 * np.pi * y) ** 2) + (y - 1) ** 2 * (
                1 + np.sin(2 * np.pi * y) ** 2)

    def derivative(self, x_k):
        x, y = x_k
        partial_x = (3 - np.cos(6 * PI * y))(x - 1) + 3 * PI * np.sin(6 * PI * x)
        partial_y = 6 * PI * (x - 1) ** 2 * np.sin(3 * PI * y) * np.cos(3 * PI * y) + \
                    4 * PI * (y - 1) ** 2 * np.sin(2 * PI * y) * np.cos(2 * PI * y) + \
                    2 * (y - 1) * (np.sin(2 * PI * y) ** 2 + 1)
        return np.array([partial_x, partial_y])

    @staticmethod
    def Hessian_Matrix(x_k):
        x, y = x_k
        partial_xx = 18 * PI * PI * np.cos(6 * PI * x) - np.cos(6 * PI * y) + 3
        partial_xy = 6 * PI * (x - 1) * np.sin(6 * PI * y)
        partial_yx = 6 * PI * (x - 1) * np.sin(6 * PI * y)
        partial_yy = 6 * PI * (x - 1) ** 2 * (-3 * PI * np.sin(3 * PI * y) ** 2 + 3 * PI * np.cos(3 * PI * y) ** 2) + \
                     4 * PI * (2 * y - 2) * np.sin(2 * PI * y) * np.cos(2 * PI * y) + \
                     4 * PI * ((y - 1) ** 2 * (- 2 * PI * np.sin(2 * PI * y) ** 2 + 2 * PI * np.cos(2 * PI * y) ** 2) +
                               2 * (y - 1) * np.sin(2 * PI * y) * np.cos(2 * PI * y)) + 2 * np.sin(2 * PI * y) ** 2 + 2

        return np.array([[partial_xx, partial_xy], [partial_yx, partial_yy]])


class HimmelblausFunction(TestFunctions2D):
    def __init__(self):
        super(HimmelblausFunction, self).__init__('Himmelblau\'s Function', -5, 5, -5, 5,
                                                  [[3, 2], [-2.805118, 3.131312],
                                                   [-3.779310, -3.283186], [3.584428, -1.848126]])

    def __call__(self, x_k):
        x, y = x_k
        return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2

    def derivative(self, x_k):
        x, y = x_k
        partial_x = 4 * x * (x ** 2 + y - 11) + 2 * x + 2 * y ** 2 - 14
        partial_y = 2 * x ** 2 + 4 * y * (x + y ** 2 - 7) + 2 * y - 22
        return np.array([partial_x, partial_y])

    @staticmethod
    def Hessian_Matrix(x_k):
        x, y = x_k
        partial_xx = 12 *x**2 + 4*y -42
        partial_xy = 4*(x+y)
        partial_yx = 4*(x+y)
        partial_yy = 4*x + 12*y**2 -26
        return np.array([[partial_xx, partial_xy], [partial_yx, partial_yy]])


class ThreeHumpCamelFunction(TestFunctions2D):
    def __init__(self):
        super(ThreeHumpCamelFunction, self).__init__('Three-Hump Camel Function', -5, 5, -5, 5, [[0, 0]])

    def __call__(self, x_k):
        x, y = x_k
        return 2 * x ** 2 - 1.05 * x ** 4 + x ** 2 / 6 + x * y + y ** 2

    def derivative(self, x_k):
        x, y = x_k
        partial_x = -21 * x ** 3 / 5 + 13 * x / 3 + y
        partial_y = x + 2 * y
        return np.array([partial_x, partial_y])

    @staticmethod
    def Hessian_Matrix(x_k):
        x, y = x_k
        partial_xx = 13/3 - 63*x**2/5
        partial_xy = 1
        partial_yx = 1
        partial_yy = 2
        return np.array([[partial_xx, partial_xy], [partial_yx, partial_yy]])


class EasomFunction(TestFunctions2D):
    def __init__(self):
        super(EasomFunction, self).__init__('Easom Function', -100, 100, -100, 100, [[np.pi, np.pi]])

    def __call__(self, x_k):
        x, y = x_k
        return -np.cos(x) * np.cos(y) * np.exp(-((x - np.pi) ** 2 + (y - np.pi) ** 2))

    def derivative(self, x_k):
        x, y = x_k
        partial_x = (-2 * (PI - x) * np.cos(x) + np.sin(x)) * np.exp(-(x - PI) ** 2 - (y - PI) ** 2) * np.cos(y)
        partial_y = (-2 * (PI - y) * np.cos(y) + np.sin(y)) * np.exp(-(x - PI) ** 2 - (y - PI) ** 2) * np.cos(x)
        return np.array([partial_x, partial_y])

    @staticmethod
    def Hessian_Matrix(x_k):
        x, y = x_k
        partial_xx = (-2*(PI-x)*(2*(PI-x)*np.cos(x)-np.sin(x)) +
                      2*(PI-x)*np.sin(x)+3*np.cos(x))*np.exp(-(x-PI)**2 - (y-PI)**2)*np.cos(y)
        partial_xy = (2*(PI-x)*np.cos(x)-np.sin(x))*(-2*(PI-y)*np.cos(y)+np.sin(y))*np.exp(-(x-PI)**2-(y-PI)**2)
        partial_yx = (2*(PI-x)*np.cos(x)+np.sin(x))*(2*(PI-y)*np.cos(y)-np.sin(y))*np.exp(-(x-PI)**2-(y-PI)**2)
        partial_yy = (-2*(PI-y)*(2*(PI-y)*np.cos(y)-np.sin(y)) +
                      2*(PI-y)*np.sin(y)+3*np.cos(y))*np.exp(-(x-PI)**2 - (y-PI)**2)*np.cos(x)
        return np.array([[partial_xx, partial_xy], [partial_yx, partial_yy]])


class McCormickFunction(TestFunctions2D):
    def __init__(self):
        super(McCormickFunction, self).__init__('McCormick Function', -1.5, 4, -3, 4, [[-0.54719, -1.54719]])

    def __call__(self, x_k):
        x, y = x_k
        return np.sin(x + y) + (x - y) ** 2 - 1.5 * x + 2.5 * y + 1

    def derivative(self, x_k):
        x, y = x_k
        partial_x = 2 * x - 2 * y + np.cos(x + y) - 1.5
        partial_y = -2 * x + 2 * y + np.cos(x + y) + 2.5
        return np.array([partial_x, partial_y])

    @staticmethod
    def Hessian_Matrix(x_k):
        x, y = x_k
        partial_xx = 2 - np.sin(x + y)
        partial_xy = -np.sin(x+y) - 2
        partial_yx = -np.sin(x+y) - 2
        partial_yy = 2 - np.sin(x + y)
        return np.array([[partial_xx, partial_xy], [partial_yx, partial_yy]])


class StyblinskiTangFunction(TestFunctions2D):
    def __init__(self):
        super(StyblinskiTangFunction, self).__init__('Styblinski-Tang Function', -5, 5, -5, 5, [[-2.903534, -2.903534]])

    def __call__(self, x_k):
        x, y = x_k
        return (x ** 4 - 16 * x ** 2 + 5 * x + y ** 4 - 16 * y ** 2 + 5 * y) / 2

    def derivative(self, x_k):
        x, y = x_k
        partial_x = 2 * x ** 3 - 16 * x + 2.5
        partial_y = 2 * y ** 3 - 16 * y + 2.5
        return np.array([partial_x, partial_y])

    @staticmethod
    def Hessian_Matrix(x_k):
        x, y = x_k
        partial_xx = 6 * x ** 2 - 16
        partial_xy = 0
        partial_yx = 0
        partial_yy = 6 * y ** 2 - 16
        return np.array([[partial_xx, partial_xy], [partial_yx, partial_yy]])


if __name__ == '__main__':
    test_function = StyblinskiTangFunction()
    test_function.save([[0, 0], [.1, .2], [.3, .4]])
