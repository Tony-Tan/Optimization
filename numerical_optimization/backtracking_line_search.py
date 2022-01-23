# algorithm 3.1 backtracking line search

import numpy as np
import sys
import test_functions.test_functions as test_f
import copy
import random


class BacktrackingLineSearch:
    def __init__(self, rho, c):
        self.rho = rho
        self.c = c

    def __call__(self, objective_function, x_k, p_k, gradient_k, alpha=0):
        if alpha == 0:
            alpha = random.random() * 2
        f_x_p = objective_function(x_k + alpha * p_k)
        f_x = objective_function(x_k)
        inc = self.c * alpha * gradient_k.dot(p_k)
        # while objective_function(x_k + alpha * p_k) > \
        #         objective_function(x_k) + \
        #         self.c * alpha * gradient_k.dot(p_k):
        while f_x_p > f_x + inc:
            alpha *= self.rho
            f_x_p = objective_function(x_k + alpha * p_k)
            inc = self.c * alpha * gradient_k.dot(p_k)
        return alpha


def steepest_descent_BLS(objective_function, threshold_delta_value,  x_init=None):
    x_sequence = []
    delta = sys.float_info.max
    x_k = np.array([random.random() * 2 - 1, random.random() * 2 - 1]) if x_init is None else x_init
    bls = BacktrackingLineSearch(rho=0.9, c=1e-4)
    f_value = objective_function(x_k)
    while threshold_delta_value < delta:
        gradient_k = objective_function.derivative(x_k)
        p_k = - gradient_k
        alpha = bls(objective_function, x_k, p_k, gradient_k, alpha=1)
        delta = f_value
        x_k += alpha * p_k
        x_sequence.append(copy.deepcopy(x_k))
        f_value = objective_function(x_k)
        delta = np.abs(delta - f_value)
    return x_sequence


if __name__ == '__main__':
    obj_f = test_f.AckleyFunction()
    x_s = steepest_descent_BLS(obj_f, 0.0001, x_init=[1.3, -1.3])
    obj_f.save(x_s)
