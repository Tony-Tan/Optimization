import numpy as np
import sys
import random
import copy
import objective_function as of


class NewtonMethod:
    def __init__(self, objective_function):
        self.obj_f = objective_function
        pass

    def run(self, threshold_delta_value, x_init=None):

        delta = sys.float_info.max
        x_k = np.array([random.random() * 2 - 1, random.random() * 2 - 1]) if x_init is None else x_init
        x_sequence = [x_k]
        f_value = self.obj_f(x_k)
        while threshold_delta_value < delta:
            gradient_k = self.obj_f.derivative(x_k)
            p_k = - np.linalg.inv(self.obj_f.hessian(x_k)).dot(gradient_k)
            delta = f_value
            x_k += p_k
            x_sequence.append(copy.deepcopy(x_k))
            f_value = self.obj_f(x_k)
            delta = np.abs(delta - f_value)
        return x_sequence


if __name__ == '__main__':
    obj_f = of.OFExample()
    sd_test = NewtonMethod(obj_f)
    x_s = sd_test.run(0.0001, x_init=[0.5, 0.])
    obj_f.draw_counter(x_s)
