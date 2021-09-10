import random

import numpy as np
import backtracking_line_search
import sys
import random
import objective_function as of
import copy


class SteepestDescent:
    def __init__(self, objective_function):
        self.obj_f = objective_function

    def run(self, threshold_delta_value, step_size=0):
        x_sequence = []
        delta = sys.float_info.max
        x_k = np.array([random.random() * 2 - 1, random.random() * 2 - 1])
        if step_size == 0:
            bls = backtracking_line_search.BacktrackingLineSearch(rho=0.9, c=1e-4)
            f_value = self.obj_f(x_k)
            while threshold_delta_value < delta:
                gradient_k = self.obj_f.derivative(x_k)
                p_k = - gradient_k
                alpha = bls(self.obj_f, x_k, p_k, gradient_k, alpha=1)
                delta = f_value
                x_k += alpha * p_k
                x_sequence.append(copy.deepcopy(x_k))
                f_value = self.obj_f(x_k)
                delta = np.abs(delta - f_value)
        else:
            alpha = step_size
            f_value = self.obj_f(x_k)
            while threshold_delta_value < delta:
                gradient_k = self.obj_f.derivative(x_k)
                p_k = - gradient_k
                delta = f_value
                x_k += alpha * p_k
                x_sequence.append(copy.deepcopy(x_k))
                f_value = self.obj_f(x_k)
                delta = np.abs(delta - f_value)
        return x_sequence


if __name__ == '__main__':
    obj_f = of.OFExample()
    sd_test = SteepestDescent(obj_f)
    x_s = sd_test.run(0.0001)
    obj_f.draw_counter(x_s)
