import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


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
        # plt.colorbar(CS)
        plt.clabel(CS, inline=True, fontsize=10)
        plt.show()


class AckleyFunction(TestFunctions2D):
    def __init__(self):
        super().__init__('Ackley Function', -5, 5, -5, 5, [[0, 0]])

    def __call__(self, x, y):
        return -20 * np.exp(-0.2 * np.sqrt(0.5 * (x ** 2 + y ** 2))) - np.exp(
            0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * x))) + np.e + 20

    def derivative(self, x, y):
        pass


class SphereFunction(TestFunctions2D):
    def __init__(self):
        super().__init__('Sphere Function', -2, 2, -2, 2, [[0, 0]])

    def __call__(self, x, y):
        return x ** 2 + y ** 2


class RosenbrockFunction(TestFunctions2D):
    def __init__(self):
        super().__init__('Sphere Function', -2, 2, -1, 3, [[1, 1]])

    def __call__(self, x, y):
        return 100 * (y - x ** 2) ** 2 + (1 - x) ** 2


class BealeFunction(TestFunctions2D):
    def __init__(self):
        super().__init__('Beale Function', -4.5, 4.5, -4.5, 4.5, [[3, 0.5]])

    def __call__(self, x, y):
        return (1.5 - x + x * y) ** 2 + (2.25 - x + x * y ** 2) ** 2 + (2.625 - x + x * y ** 3) ** 2


class GoldsteinPriceFunction(TestFunctions2D):
    def __init__(self):
        super().__init__('Goldstein Price Function', -2, 2, -2, 2, [[0, -1]])

    def __call__(self, x, y):
        return (1 + (x + y + 1) ** 2 * (19 - 14 * x + 3 * x ** 2 - 14 * y + 6 * x * y + 3 * y ** 2)) * (
                    30 + (2 * x - 3 * y) ** 2 * (18 - 32 * x + 12 * x ** 2 + 48 * y - 36 * x * y + 27 * y ** 2))


class BoothFunction(TestFunctions2D):
    def __init__(self):
        super().__init__('Booth Function', -10, 10, -10, 10, [[1, 3]])

    def __call__(self, x, y):
        return (x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2


class MatyasFunction(TestFunctions2D):
    def __init__(self):
        super(MatyasFunction, self).__init__('Matyas Function', -10, 10, -10, 10, [[0, 0]])

    def __call__(self, x, y):
        return 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y


class LeviFunctionN13(TestFunctions2D):
    def __init__(self):
        super(LeviFunctionN13, self).__init__('Levi Function N. 13', -10, 10, -10, 10, [[1, 1]])

    def __call__(self, x, y):
        return np.sin(3 * np.pi * x) ** 2 + (x - 1) ** 2 * (1 + np.sin(3 * np.pi * y) ** 2) + (y - 1) ** 2 * (
                    1 + np.sin(2 * np.pi * y) ** 2)


class HimmelblausFunction(TestFunctions2D):
    def __init__(self):
        super(HimmelblausFunction, self).__init__('Himmelblau\'s Function', -5, 5, -5, 5,
                                                  [[3, 2], [-2.805118, 3.131312],
                                                   [-3.779310, -3.283186], [3.584428, -1.848126]])

    def __call__(self, x, y):
        return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2


class Three_HumpCamelFunction(TestFunctions2D):
    def __init__(self):
        super(Three_HumpCamelFunction, self).__init__('Three-Hump Camel Function', -5, 5, -5, 5, [[0, 0]])

    def __call__(self, x, y):
        return 2 * x ** 2 - 1.05 * x ** 4 + x ** 2 / 6 + x * y + y ** 2


class EasomFunction(TestFunctions2D):
    def __init__(self):
        super(EasomFunction, self).__init__('Easom Function', -100, 100, -100, 100, [[np.pi, np.pi]])

    def __call__(self, x, y):
        return -np.cos(x) * np.cos(y) * np.exp(-((x - np.pi) ** 2 + (y - np.pi) ** 2))


class McCormickFunction(TestFunctions2D):
    def __init__(self):
        super(McCormickFunction, self).__init__('McCormick Function', -1.5, 4, -3, 4, [[-0.54719, -1.54719]])

    def __call__(self, x, y):
        return np.sin(x + y) + (x - y) ** 2 - 1.5 * x + 2.5 * y + 1


class StyblinskiTangFunction(TestFunctions2D):
    def __init__(self):
        super(StyblinskiTangFunction, self).__init__('Styblinski-Tang Function',-5,5,-5,5,[[-2.903534,-2.903534]])

    def __call__(self, x, y):
        return (x**4-16*x**2+5*x + y**4 -16*y**2 + 5*y)/2



if __name__ == '__main__':
    test_function = StyblinskiTangFunction()
    test_function.show([[0, 0], [.1, .2], [.3, .4]])
