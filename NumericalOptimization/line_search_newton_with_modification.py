import numpy as np
import sys
import random
import copy
import objective_function as of
import backtracking_line_search


class LineSearchNewtonMethod:
    def __init__(self, objective_function):
        self.obj_f = objective_function
        pass

    def run(self, threshold_delta_value, x_init=None):
        hessian_modify_delta = 1
        delta = sys.float_info.max
        x_k = np.array([random.random() * 4 - 2, random.random() * 4 - 2]) if x_init is None else x_init
        x_sequence = [copy.deepcopy(x_k)]
        f_value = self.obj_f(x_k)
        bls = backtracking_line_search.BacktrackingLineSearch(rho=0.9, c=1e-4)
        while threshold_delta_value < delta:
            gradient_k = self.obj_f.derivative(x_k)
            hessian_x_k = self.obj_f.hessian(x_k)
            # negative definite Hessian detect
            eig_value, eig_vectors = np.linalg.eig(hessian_x_k)
            min_eig_value = eig_value.min()
            if min_eig_value < 0:
                print("negative eigenvalue detected and modify the Hessian")
                hessian_x_k += (-min_eig_value+hessian_modify_delta)*np.ones(2)
            p_k = - np.linalg.inv(hessian_x_k).dot(gradient_k)
            alpha = bls(self.obj_f, x_k, p_k, gradient_k, alpha=1)
            delta = f_value
            x_k += alpha * p_k
            x_sequence.append(copy.deepcopy(x_k))
            f_value = self.obj_f(x_k)
            delta = np.abs(delta - f_value)
        return x_sequence


if __name__ == '__main__':
    obj_f = of.OFExample()
    sd_test = LineSearchNewtonMethod(obj_f)
    x_s = sd_test.run(0.0001, x_init=[0, 0.4])
    obj_f.draw_counter(x_s)
