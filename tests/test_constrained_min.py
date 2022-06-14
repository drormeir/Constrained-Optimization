import unittest
from src.constrained_min import LineSearch
from src.utils import *
from examples import *


class TestConstrainedMin(unittest.TestCase):

    def test_qp(self):
        qp = QuadraticFunction(q2=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], q1=[0, 0, 2], q0=1)  # x^2+y^2+(z+1)^2
        qp_eq_mat, qp_eq_rhs = [[1, 1, 1]], 1  # x + y + z == 1
        qp_ineq1 = QuadraticFunction(q1=[-1, 0, 0])  # x >= 0
        qp_ineq2 = QuadraticFunction(q1=[0, -1, 0])  # y >= 0
        qp_ineq3 = QuadraticFunction(q1=[0, 0, -1])  # z >= 0
        ls = LineSearch(verbose=True)
        res = ls.interior_pt(func=qp, x0=(0.1, 0.2, 0.7), is_minimize=True,
                             ineq_constraints=[qp_ineq1, qp_ineq2, qp_ineq3], eq_constraints_mat=qp_eq_mat,
                             eq_constraints_rhs=qp_eq_rhs, print_every=1)
        self.assertTrue(res['success'])
        print(res, flush=True)
        func_name = "Quadratic problem: Minimize x^2+y^2+(z+1)^2"
        plot_line_search(func_name, ls)

    def test_lp(self):
        lp = QuadraticFunction(q1=[1, 1])
        lp_ineq1 = QuadraticFunction(q1=[-1, -1], q0=1)  # y >= -x + 1
        lp_ineq2 = QuadraticFunction(q1=[0, 1], q0=-1)  # y <= 1
        lp_ineq3 = QuadraticFunction(q1=[1, 0], q0=-2)  # x <= 2
        lp_ineq4 = QuadraticFunction(q1=[0, -1], q0=0)  # y >= 0
        ls = LineSearch(verbose=True)
        res = ls.interior_pt(func=lp, x0=(0.5, 0.75), is_minimize=False,
                             ineq_constraints=[lp_ineq1, lp_ineq2, lp_ineq3, lp_ineq4],
                             print_every=1)
        self.assertTrue(res['success'])
        print(res, flush=True)
        func_name = "Linear problem: Maximize x+y"
        plot_line_search(func_name, ls)

