import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

PI = np.pi


class TestFunctions2D:
    def __init__(self, function_name, x_min, x_max, y_min, y_max, global_minimum):
        self.name = function_name
        self.search_domain = {'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max}
        self.search_range = {'x_range': x_max - x_min, 'y_range': y_max - y_min}
        self.global_minimum = global_minimum

    def __call__(self, *args):
        return 0

    def derivative(self, *args):
        return 0

    def show(self, trajectory):
        x = np.arange(self.search_domain['x_min'], self.search_domain['x_max'], self.search_range['x_range'] / 1000.)
        y = np.arange(self.search_domain['y_min'], self.search_domain['y_max'], self.search_range['y_range'] / 1000.)
        X, Y = np.meshgrid(x, y)
        Z = self.__call__(X, Y)
        plt.figure(figsize=(10, 10))
        CS = plt.contour(X, Y, Z, 40, alpha=0.7, cmap=mpl.cm.jet)
        for global_min_i in self.global_minimum:
            plt.scatter(global_min_i[0], global_min_i[1], c='r', marker='o')
        for position_i in trajectory:
            plt.scatter(position_i[0], position_i[1], c='b', marker='x', alpha=0.7)
        plt.clabel(CS, inline=True, fontsize=10)
        plt.show()


class AckleyFunction(TestFunctions2D):
    def __init__(self):
        super().__init__('Ackley Function', -5, 5, -5, 5, [[0, 0]])

    def __call__(self, x, y):
        return -20 * np.exp(-0.2 * np.sqrt(0.5 * (x ** 2 + y ** 2))) - np.exp(
            0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) + np.e + 20

    def derivative(self, x, y):
        part_1 = (2 * np.sqrt(2) * np.exp(-(np.sqrt(2 * (x ** 2 + y ** 2))) / 10)) / (np.sqrt(x ** 2 + y ** 2))
        part_2 = PI * np.exp(np.cos(2 * PI * x) / 2 + np.cos(2 * PI * y) / 2)
        partial_x = part_1 * x + part_2 * np.sin(2 * PI * x)
        partial_y = part_1 * y + part_2 * np.sin(2 * PI * y)
        return np.array([partial_x, partial_y])


class SphereFunction(TestFunctions2D):
    def __init__(self):
        super().__init__('Sphere Function', -2, 2, -2, 2, [[0, 0]])

    def __call__(self, x, y):
        return x ** 2 + y ** 2

    def derivative(self, x, y):
        partial_x = 2 * x
        partial_y = 2 * y
        return np.array([partial_x, partial_y])


class RosenbrockFunction(TestFunctions2D):
    def __init__(self):
        super().__init__('Sphere Function', -2, 2, -1, 3, [[1, 1]])

    def __call__(self, x, y):
        return 100 * (y - x ** 2) ** 2 + (1 - x) ** 2

    def derivative(self, x, y):
        partial_x = -400 * x * (y - x ** 2) + 2 * x - 2
        partial_y = -200 * x ** 2 + 200 * y
        return np.array([partial_x, partial_y])


class BealeFunction(TestFunctions2D):
    def __init__(self):
        super().__init__('Beale Function', -4.5, 4.5, -4.5, 4.5, [[3, 0.5]])

    def __call__(self, x, y):
        return (1.5 - x + x * y) ** 2 + (2.25 - x + x * y ** 2) ** 2 + (2.625 - x + x * y ** 3) ** 2

    def derivative(self, x, y):
        partial_x = 0.25 * (y - 1) * (
                    8 * x * y ** 5 + 8 * x * y ** 4 + 16 * x * y ** 3 - 8 * x * y - 24 * x + 21 * y ** 2 + 39 * y + 51)
        partial_y = 0.25 * x * (
                    24 * x * y ** 5 + 16 * x * y ** 3 - 24 * x * y ** 2 - 8 * x * y - 8 * x + 63 * y ** 2 + 36 * y + 12)
        return np.array([partial_x, partial_y])


class GoldsteinPriceFunction(TestFunctions2D):
    def __init__(self):
        super().__init__('Goldstein Price Function', -2, 2, -2, 2, [[0, -1]])

    def __call__(self, x, y):
        return (1 + (x + y + 1) ** 2 * (19 - 14 * x + 3 * x ** 2 - 14 * y + 6 * x * y + 3 * y ** 2)) * (
                30 + (2 * x - 3 * y) ** 2 * (18 - 32 * x + 12 * x ** 2 + 48 * y - 36 * x * y + 27 * y ** 2))

    def derivative(self, x, y):
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


class BoothFunction(TestFunctions2D):
    def __init__(self):
        super().__init__('Booth Function', -10, 10, -10, 10, [[1, 3]])

    def __call__(self, x, y):
        return (x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2

    def derivative(self, x, y):
        partial_x = 10 * x + 8 * y - 34
        partial_y = 8 * x + 10 * y - 38
        return np.array([partial_x, partial_y])


class MatyasFunction(TestFunctions2D):
    def __init__(self):
        super(MatyasFunction, self).__init__('Matyas Function', -10, 10, -10, 10, [[0, 0]])

    def __call__(self, x, y):
        return 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y

    def derivative(self, x, y):
        partial_x = (13 * x - 12 * y) / 25
        partial_y = -12 * x / 25 + 13 * y / 25
        return np.array([partial_x, partial_y])


class LeviFunctionN13(TestFunctions2D):
    def __init__(self):
        super(LeviFunctionN13, self).__init__('Levi Function N. 13', -10, 10, -10, 10, [[1, 1]])

    def __call__(self, x, y):
        return np.sin(3 * np.pi * x) ** 2 + (x - 1) ** 2 * (1 + np.sin(3 * np.pi * y) ** 2) + (y - 1) ** 2 * (
                1 + np.sin(2 * np.pi * y) ** 2)

    def derivative(self, x, y):
        partial_x = (3 - np.cos(6 * PI * y))(x - 1) + 3 * PI * np.sin(6 * PI * x)
        partial_y = 6 * PI * (x - 1) ** 2 * np.sin(3 * PI * y) * np.cos(3 * PI * y) + \
                    4 * PI * (y - 1) ** 2 * np.sin(2 * PI * y) * np.cos(2 * PI * y) + \
                    2 * (y - 1) * (np.sin(2 * PI * y) ** 2 + 1)
        return np.array([partial_x, partial_y])


class HimmelblausFunction(TestFunctions2D):
    def __init__(self):
        super(HimmelblausFunction, self).__init__('Himmelblau\'s Function', -5, 5, -5, 5,
                                                  [[3, 2], [-2.805118, 3.131312],
                                                   [-3.779310, -3.283186], [3.584428, -1.848126]])

    def __call__(self, x, y):
        return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2

    def derivative(self, x, y):
        partial_x = 4 * x * (x ** 2 + y - 11) + 2 * x + 2 * y ** 2 - 14
        partial_y = 2 * x ** 2 + 4 * y * (x + y ** 2 - 7) + 2 * y - 22
        return np.array([partial_x, partial_y])


class Three_HumpCamelFunction(TestFunctions2D):
    def __init__(self):
        super(Three_HumpCamelFunction, self).__init__('Three-Hump Camel Function', -5, 5, -5, 5, [[0, 0]])

    def __call__(self, x, y):
        return 2 * x ** 2 - 1.05 * x ** 4 + x ** 2 / 6 + x * y + y ** 2

    def derivative(self, x, y):
        partial_x = -21 * x ** 3 / 5 + 13 * x / 3 + y
        partial_y = x + 2 * y
        return np.array([partial_x, partial_y])


class EasomFunction(TestFunctions2D):
    def __init__(self):
        super(EasomFunction, self).__init__('Easom Function', -100, 100, -100, 100, [[np.pi, np.pi]])

    def __call__(self, x, y):
        return -np.cos(x) * np.cos(y) * np.exp(-((x - np.pi) ** 2 + (y - np.pi) ** 2))

    def derivative(self, x, y):
        partial_x = (-2 * (PI - x) * np.cos(x) + np.sin(x)) * np.exp(-(x - PI) ** 2 - (y - PI) ** 2) * np.cos(y)
        partial_y = (-2 * (PI - y) * np.cos(y) + np.sin(y)) * np.exp(-(x - PI) ** 2 - (y - PI) ** 2) * np.cos(x)
        return np.array([partial_x, partial_y])


class McCormickFunction(TestFunctions2D):
    def __init__(self):
        super(McCormickFunction, self).__init__('McCormick Function', -1.5, 4, -3, 4, [[-0.54719, -1.54719]])

    def __call__(self, x, y):
        return np.sin(x + y) + (x - y) ** 2 - 1.5 * x + 2.5 * y + 1

    def derivative(self, x, y):
        partial_x = 2 * x - 2 * y + np.cos(x + y) - 1.5
        partial_y = -2 * x + 2 * y + np.cos(x + y) + 2.5
        return np.array([partial_x, partial_y])


class StyblinskiTangFunction(TestFunctions2D):
    def __init__(self):
        super(StyblinskiTangFunction, self).__init__('Styblinski-Tang Function', -5, 5, -5, 5, [[-2.903534, -2.903534]])

    def __call__(self, x, y):
        return (x ** 4 - 16 * x ** 2 + 5 * x + y ** 4 - 16 * y ** 2 + 5 * y) / 2

    def derivative(self, x, y):
        partial_x = 2 * x ** 3 - 16 * x + 2.5
        partial_y = 2 * y ** 3 - 16 * y + 2.5
        return np.array([partial_x, partial_y])


if __name__ == '__main__':
    test_function = StyblinskiTangFunction()
    test_function.show([[0, 0], [.1, .2], [.3, .4]])
