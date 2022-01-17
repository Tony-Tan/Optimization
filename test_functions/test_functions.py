import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


class TestFunctions2D:
    def __init__(self, function_name, x_min, x_max, y_min, y_max):
        self.name = function_name
        self.search_domain = {'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max}
        self.search_range = {'x_range': x_max - x_min, 'y_range': y_max - y_min}
        self.global_minimum = 0

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
        for position_i in trajectory:
            plt.scatter(position_i[0], position_i[1], c='b', marker='x', alpha=0.7)
        plt.colorbar(CS)
        plt.show()


class AckleyFunction(TestFunctions2D):
    def __init__(self):
        super().__init__('Ackley Function', -5, 5, -5, 5)

    def __call__(self, x, y):
        return -20 * np.exp(-0.2 * np.sqrt(0.5 * (x ** 2 + y ** 2))) - np.exp(
            0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * x))) + np.e + 20

    def derivative(self, x, y):
        pass


class SphereFunction(TestFunctions2D):
    def __init__(self):
        super().__init__('Sphere Function', -2, 2, -2, 2)

    def __call__(self, x, y):
        return x ** 2 + y ** 2


class RosenbrockFunction(TestFunctions2D):
    def __init__(self):
        super().__init__('Sphere Function', -2, 2, -1, 3)

    def __call__(self, x, y):
        return 100*(y-x**2)**2 + (1-x)**2


if __name__ == '__main__':
    test_function = RosenbrockFunction()
    test_function.show([[0, 0], [.1, .2], [.3, .4]])
