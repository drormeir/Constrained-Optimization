import math
import numpy as np


class QuadraticFunction:
    def __init__(self, q2=None, q1=None, q0=0):
        self.q0 = q0
        if q2 is None:
            # linear function
            self.q1 = np.array(q1).reshape(-1, 1)
            self.dim = self.q1.shape[0]
            self.q2 = np.zeros(shape=(self.dim, self.dim))
        elif q1 is None:
            # quadratic function without linear term
            self.q2 = np.array(q2)
            self.dim = q2.shape[0]
            self.q1 = np.zeros(shape=(self.dim, 1))
        else:
            self.q1 = np.array(q1).reshape(-1, 1)
            self.dim = len(q1)
            self.q2 = np.array(q2)
        self.hessian = self.q2 + self.q2.T

    def debug_print(self):
        print("QuadraticFunction: dim={} Q0={} Q1={}\nQ2={}\n".format(self.dim, self.q0, self.q1.reshape(-1), self.q2))

    def __call__(self, x, calc_gradient=True, calc_hessian=False):
        # x might be a matrix of vertical vectors of (x1,x2)
        # f = (x.T.dot(self.q2).dot(x) + self.q1.T.dot(x) + self.q0).item()
        f = np.einsum("ij,ji->i", x.T.dot(self.q2), x) + self.q1.T.dot(x) + self.q0
        if f.size == 1:
            f = f.item()
        if not calc_gradient:
            return f
        g = self.hessian.dot(x) + self.q1
        h = self.hessian if calc_hessian else None
        return f, g, h


quad_func1 = QuadraticFunction(q2=np.array([[1, 0], [0, 1]], dtype=float))
q1_1000 = np.array([[1, 0], [0, 1000]], dtype=float)
quad_func2 = QuadraticFunction(q2=q1_1000)
q_rotate = np.array([[math.sqrt(3) / 2, -0.5], [0.5, math.sqrt(3) / 2]], dtype=float)
quad_func3 = QuadraticFunction(q2=q_rotate.T.dot(q1_1000).dot(q_rotate))

linear_function = QuadraticFunction(q1=[1, 1])


def rosenbrock(x, calc_gradient=True, calc_hessian=False):
    # x might be a matrix of vertical vectors of (x1,x2)
    x1, x2 = x[0], x[1]
    f = 100 * (x2 - x1 * x1) ** 2 + (1 - x1) ** 2
    if f.size == 1:
        f = f.item()
    if not calc_gradient:
        return f
    g = np.vstack([400 * x1 ** 3 - 400 * x1 * x2 + 2 * x1 - 2, 200 * (x2 - x1 * x1)])
    assert g.shape == x.shape
    if calc_hessian and x.shape[1] == 1:
        x1, x2 = x1.item(), x2.item()
        h = np.array([[1200 * x1 * x1 - 400 * x2 + 2, -400 * x1], [-400 * x1, 200]], dtype=float)
    else:
        h = None
    return f, g, h


def boyd(x, calc_gradient=True, calc_hessian=False):
    # x might be a matrix of vertical vectors of (x1,x2)
    x1, x2 = x[0], x[1]
    f1 = np.exp(x1 + 3 * x2 - 0.1)
    f2 = np.exp(x1 - 3 * x2 - 0.1)
    f3 = np.exp(-x1 - 0.1)
    f = f1 + f2 + f3
    if f.size == 1:
        f = f.item()
    if not calc_gradient:
        return f
    g2 = 3 * f1 - 3 * f2
    g = np.vstack([f1 + f2 - f3, g2])
    assert g.shape == x.shape
    if calc_hessian and x.shape[1] == 1:
        f1 = f1.item()
        f2 = f2.item()
        g2 = g2.item()
        h = np.array([[f, g2], [g2, 9 * f1 + 9 * f2]])
    else:
        h = None
    return f, g, h
