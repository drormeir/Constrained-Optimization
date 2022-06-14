import unittest
from src.constrained_min import LineSearch
from src.utils import *
from examples import *


class TestLineSearch(unittest.TestCase):

    def test_quad1(self):
        self.run_quad_test(quad_func1, 1)

    def test_quad2(self):
        self.run_quad_test(quad_func2, 2)

    def test_quad3(self):
        self.run_quad_test(quad_func3, 3)

    def run_quad_test(self, q_func, q_num):
        ls = [self.run_quad(f=q_func, q_num=q_num, newton=False, step_len=1e-4, wolfe_c1=0),
              self.run_quad(f=q_func, q_num=q_num, newton=False, step_len=0.1, wolfe_c1=0.01),
              self.run_quad(f=q_func, q_num=q_num, newton=True, wolfe_c1=0),
              self.run_quad(f=q_func, q_num=q_num, newton=True, wolfe_c1=0.01)]
        func_name: str = "Quadratic function {}. Q=\n{}".format(q_num, q_func.q2)
        plot_line_search(func_name, ls)

    def run_quad(self, f, q_num, newton, step_len=None, wolfe_c1=0.):
        ls = LineSearch(verbose=True, wolfe_c1=wolfe_c1, newton=newton, max_iter=100000)
        opt_type = "Newton Method" if newton else "Gradient Descent"
        wolfe_text = " Wolfe" if wolfe_c1 > 0 else ""
        print("\nMinimizing quadratic function {} with {}{}:\nQ={}\n"
              .format(q_num, opt_type, wolfe_text, f.q2) + "=" * 50)
        res = ls.unconstrained_minimize(f=f, x0=[1, 1], step_len=step_len, print_every=1 if newton else None)
        print(res, flush=True)
        self.assertLess(res['objective'], 1e-6)
        self.assertLess(np.linalg.norm(res['location']), 1e-4)
        if newton:
            self.assertEqual(res['num_iter'], 2)
        return ls

    def test_linear(self):
        ls = [self.run_linear(newton=False, step_len=0.1, wolfe_c1=0.01),
              self.run_linear(newton=True, wolfe_c1=0.01)]
        func_name = "Linear function.\na={}".format(linear_function.a.reshape(-1))

        plot_line_search(func_name, ls)

    def run_linear(self, newton, step_len=None, wolfe_c1=0.):
        ls = LineSearch(verbose=True, newton=newton, wolfe_c1=wolfe_c1)
        opt_type = "Newton Method" if newton else "Gradient Descent"
        wolfe_text = " Wolfe" if wolfe_c1 > 0 else ""
        print("\nMinimizing linear function with {}{}:\na={}\n"
              .format(opt_type, wolfe_text, linear_function.a.reshape(-1)) + "=" * 50)
        res1 = ls.unconstrained_minimize(f=linear_function, x0=[1, 1], step_len=step_len)
        print(res1, flush=True)
        self.assertEqual(res1['num_iter'], 1 if newton else ls.max_iter)
        self.assertFalse(res1['success'])
        return ls

    def test_rosenbrock(self):
        ls = [self.run_rosenbrock(newton=False, step_len=1e-3, wolfe_c1=0),
              self.run_rosenbrock(newton=False, step_len=1e-2, wolfe_c1=0.01),
              self.run_rosenbrock(newton=True, wolfe_c1=0),
              self.run_rosenbrock(newton=True, wolfe_c1=0.01)]
        func_name = "Rosenbrock function"
        plot_line_search(func_name, ls)

    def run_rosenbrock(self, newton, step_len=None, wolfe_c1=0.0):
        x0_rosenbrock = [-1, 2]
        target_rosenbrock = np.array([1, 1], dtype=float)
        ls = LineSearch(verbose=True, max_iter=100000)
        opt_type = "Newton Method" if newton else "Gradient Descent"
        wolfe_text = " Wolfe" if wolfe_c1 > 0 else ""
        print("\nMinimizing Rosenbrock with {}{}:\n".format(opt_type, wolfe_text) + "=" * 50)
        res = ls.unconstrained_minimize(f=rosenbrock, x0=x0_rosenbrock, step_len=step_len, wolfe_c1=wolfe_c1,
                                        newton=newton)
        print(res, flush=True)
        if newton and wolfe_c1 < 1e-6:
            self.assertLessEqual(res['num_iter'], 2)
        else:
            self.assertLess(res['objective'], 1e-6)
            self.assertLess(np.linalg.norm(res['location'] - target_rosenbrock), 1e-3)
        return ls

    def test_boyd(self):
        ls = [self.run_boyd(newton=False, step_len=1e-3, wolfe_c1=0),
              self.run_boyd(newton=False, step_len=0.1, wolfe_c1=0.01),
              self.run_boyd(newton=True, wolfe_c1=0),
              self.run_boyd(newton=True, wolfe_c1=0.01)]
        func_name = "Boyd function."
        plot_line_search(func_name, ls)

    def run_boyd(self, newton, step_len=None, wolfe_c1=0.0):
        x0_boyd = [1, 1]
        target_boyd = np.array([0.5 * math.log(0.5), 0], dtype=float)
        objective_boyd = math.exp(-0.1) * 2 * math.sqrt(2)
        ls = LineSearch(verbose=True, max_iter=100000, print_every=1)
        opt_type = "Newton Method" if newton else "Gradient Descent"
        wolfe_text = " Wolfe" if wolfe_c1 > 0 else ""
        print("\nMinimizing Boyd with {}{}:\n".format(opt_type, wolfe_text) + "=" * 50)
        res = ls.unconstrained_minimize(f=boyd, x0=x0_boyd, newton=newton, step_len=step_len, wolfe_c1=wolfe_c1)
        print(res, flush=True)
        self.assertLess(np.linalg.norm(res['location'] - target_boyd), 2e-5)
        self.assertLess(abs(res['objective'] - objective_boyd), 1e-6)
        return ls


if __name__ == '__main__':
    unittest.main()
