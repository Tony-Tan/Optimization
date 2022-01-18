# algorithm 3.1 backtracking line search
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


if __name__ == '__main__':
    bls = BacktrackingLineSearch(rho=0.7, c=0.5)
    OF_Q = of.Quadratic

