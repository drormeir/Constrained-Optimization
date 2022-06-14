import math
import numpy as np


class Lagrangian:
    def __init__(self, f, is_minimize, A=None, b=None, inequalities=None, t=1):
        for x_dim in range(1, 1000):
            try:
                f(np.zeros(shape=(x_dim, 1)), calc_gradient=False)
                self.x_dim = x_dim
                break
            except ValueError:
                pass
        self.f = f
        if (isinstance(A, np.ndarray) and A.size > 0) or (isinstance(A, list) and A):
            self.equality_constrains = True
            self.A = np.array(A)
            self.b = np.array(b).reshape(-1, 1)
            self.nu_dim = self.b.shape[0]  # number of equality constraints
            assert self.A.shape == (self.nu_dim, self.x_dim)
        else:
            self.equality_constrains = False
            self.nu_dim = 0
        self.nu_hessian = np.zeros(shape=(self.nu_dim, self.nu_dim))
        if isinstance(inequalities, list) and inequalities:
            self.inequalities = inequalities
            self.t = float(t)
        else:
            self.inequalities = []
            self.t = 1.0
        self.is_minimize = 1 if is_minimize else -1

    def __call__(self, x, calc_gradient=True, calc_hessian=False):
        x, nu = x[:self.x_dim], x[self.x_dim:]
        f_res = self.f(x, calc_gradient, calc_hessian)
        factor = self.t * self.is_minimize
        lagrange_f = factor * (f_res[0] if calc_gradient else f_res)
        if self.equality_constrains:
            lagrange_f += nu.T.dot(self.A.dot(x) - self.b).reshape(-1)
        inequalities_res = [ineq(x, calc_gradient, calc_hessian) for ineq in self.inequalities]
        if not calc_gradient:
            if inequalities_res:
                inequalities_f = np.array(inequalities_res)
                if inequalities_f.max() > -np.finfo(float).eps:
                    return np.inf
                lagrange_f -= np.log(-inequalities_f).sum()
            return lagrange_f
        lagrange_g = factor * f_res[1]
        lagrange_h = factor * f_res[2] if calc_hessian else None
        if inequalities_res:
            inequalities_f = np.array([ineq[0] for ineq in inequalities_res])
            if inequalities_f.max() > -np.finfo(float).eps:
                return np.inf, None, None
            inequalities_g = np.hstack([ineq[1] for ineq in inequalities_res]) / inequalities_f
            if calc_hessian:
                inequalities_hh = np.sum(
                    np.array([ineq[2] for ineq in inequalities_res]) / inequalities_f.reshape((-1, 1, 1)), axis=0)
                inequalities_hg = np.einsum("ij,kj->ik", inequalities_g, inequalities_g)  # sum (g*g.T)
                inequalities_h = inequalities_hh - inequalities_hg
                lagrange_h -= inequalities_h
            lagrange_g -= np.sum(inequalities_g, axis=1).reshape(-1, 1)
            lagrange_f -= np.log(-inequalities_f).sum()

        if self.equality_constrains:
            lagrange_g = np.vstack([lagrange_g + self.A.T.dot(nu), self.A.dot(x) - self.b])
            if calc_hessian:
                lagrange_h = np.vstack([np.hstack([lagrange_h, self.A.T]), np.hstack([self.A, self.nu_hessian])])
        if isinstance(lagrange_f, np.ndarray):
            if len(lagrange_f) == 1:
                lagrange_f = lagrange_f.item()
        elif isinstance(lagrange_f, list):
            if len(lagrange_f) == 1:
                lagrange_f = lagrange_f[0]
        return lagrange_f, lagrange_g, lagrange_h

    def f_inner(self, x_nu):
        assert len(x_nu) >= self.x_dim
        return self.is_minimize * self.f(x_nu[:self.x_dim], calc_gradient=False)

    def obj_tol(self):
        return len(self.inequalities) / self.t

    def get_x0(self, x0):
        return np.vstack([np.array(x0).reshape(-1, 1), np.zeros(shape=(self.nu_dim, 1))])

    def eq_total_norm(self, x_nu):
        if not self.equality_constrains:
            return 0
        assert len(x_nu) >= self.x_dim
        return np.linalg.norm(self.A.dot(x_nu[:self.x_dim]) - self.b)

    def eq_constrains_values(self, x_nu):
        if not self.equality_constrains:
            return []
        assert len(x_nu) >= self.x_dim
        return list((self.A.dot(x_nu[:self.x_dim]) - self.b).reshape(-1))

    def ineq_constrains_values(self, x_nu):
        assert len(x_nu) >= self.x_dim
        return [ineq(x_nu[:self.x_dim], calc_gradient=False) for ineq in self.inequalities]

    def feasible_points(self, points, eps):
        res = np.full(fill_value=True, shape=(points.shape[1],))
        for ineq in self.inequalities:
            val = ineq(points, calc_gradient=False).reshape(-1)
            res &= val < eps
        return res


class MinimizerLoopTracker:
    def __init__(self, obj_tol, param_tol, verbose, print_every, max_iter, location_dim=None, is_minimize=True):
        self.objective_values = []
        self.location_values = []
        self.obj_tol = obj_tol
        self.param_tol2 = param_tol * param_tol
        self.verbose = verbose
        self.print_every = print_every
        self.max_iter = max_iter
        self.success = False
        self.nu_values = None
        self.location_dim = location_dim
        self.is_minimize = is_minimize

    def append(self, f_val, location):
        if not self.objective_values:
            self.location_values = [location.copy()]
            self.objective_values = [f_val]
            if self.verbose and (self.print_every == 1):
                print("Iter[0]: objective={} location: {}".format(self.objective_values[-1],
                                                                  self.location_values[-1].reshape(-1)))
            return True
        objective_improvement = self.objective_values[-1] - f_val
        if objective_improvement > 0:
            loop_iter = len(self.objective_values)
            self.objective_values.append(f_val)
            self.location_values.append(location.copy())
            if self.verbose and ((loop_iter % self.print_every == 0) or (loop_iter == self.max_iter)):
                print("Iter[{}]: objective={} location: {}".format(loop_iter, self.objective_values[-1],
                                                                   self.location_values[-1].reshape(-1)))
        if objective_improvement < self.obj_tol:
            self.success = True
            if self.verbose:
                print("Optimization reached the target of objective function.")
            return False
        p_step = location - self.location_values[-2]
        if p_step.T.dot(p_step).item() < self.param_tol2:
            self.success = True
            if self.verbose:
                print("Optimization reached the target of location coordinates.")
            return False
        return True

    def summary(self):
        self.objective_values = np.array(self.objective_values)
        if not self.is_minimize:
            self.objective_values *= -1
        self.location_values = np.hstack(self.location_values)
        res = {'objective': self.objective_values[-1],
               'success': self.success,
               'num_iter': len(self.objective_values) - 1}
        if self.location_dim is not None:
            self.nu_values = self.location_values[self.location_dim:]
            self.location_values = self.location_values[:self.location_dim]
            res['final_nu'] = self.nu_values[:, -1].reshape(-1) if self.nu_values.size > 0 else None
            res['dim'] = self.location_dim
        else:
            res['dim'] = self.location_values.shape[0]
        res['location'] = self.location_values[:, -1].reshape(-1)
        norms2 = np.einsum("ij,ij->j", self.location_values, self.location_values)
        res['max_norm_location'] = math.sqrt(max(norms2))
        res['min_objective_value'] = self.objective_values.min()
        res['max_objective_value'] = self.objective_values.max()
        return self.objective_values, self.location_values, res


class LineSearch:
    eps = np.finfo(float).eps
    max_cond = 1 / np.finfo(float).eps

    def __init__(self, step_len=0.1, obj_tol=1e-12, param_tol=1e-8, max_iter=100, newton=False, wolfe_c1=0.0,
                 wolfe_backtracking=0.5, verbose=True, print_every=5, max_iter_outer_lagrangian=100, mu_lagrangian=10,
                 t0_lagrangian=1):
        self.step_len = step_len
        self.obj_tol = obj_tol
        self.param_tol = param_tol
        self.max_iter = max_iter
        self.newton = newton
        self.verbose = verbose
        self.print_every = print_every
        self.wolfe_c1 = wolfe_c1
        self.wolfe_backtracking = wolfe_backtracking
        self.run_mode_label = ""
        self.res = {}
        self.location_values = None
        self.objective_values = []
        self.max_iter_outer_lagrangian = max_iter_outer_lagrangian
        self.mu_lagrangian = mu_lagrangian
        self.t0_lagrangian = t0_lagrangian
        self.run_param = {}
        self.save_function = None

    def __len__(self):
        return len(self.objective_values)

    def interior_pt(self, func, x0, is_minimize,
                    ineq_constraints=None, eq_constraints_mat=None, eq_constraints_rhs=None,
                    obj_tol=None, param_tol=None, max_iter=None,
                    wolfe_c1=None, wolfe_backtracking=None, verbose=None,
                    print_every=None, max_iter_outer_lagrangian=None, mu_lagrangian=None, t0_lagrangian=None):
        if wolfe_c1 is None:
            wolfe_c1 = max(self.wolfe_c1, 0.01)
        self.set_run_params(newton=True, wolfe_c1=wolfe_c1, wolfe_backtracking=wolfe_backtracking, force=True)
        if obj_tol is None:
            obj_tol = self.obj_tol
        if param_tol is None:
            param_tol = self.param_tol
        if verbose is None:
            verbose = self.verbose
        if print_every is None:
            print_every = self.print_every
        if max_iter_outer_lagrangian is None:
            max_iter_outer_lagrangian = self.max_iter_outer_lagrangian
        if mu_lagrangian is None:
            mu_lagrangian = self.mu_lagrangian
        if t0_lagrangian is None:
            t0_lagrangian = self.t0_lagrangian
        lagrangian = Lagrangian(func, A=eq_constraints_mat, b=eq_constraints_rhs,
                                inequalities=ineq_constraints,
                                t=t0_lagrangian, is_minimize=is_minimize)
        loop_tracker = MinimizerLoopTracker(obj_tol=obj_tol, param_tol=param_tol, verbose=verbose,
                                            print_every=print_every,
                                            max_iter=max_iter_outer_lagrangian, location_dim=lagrangian.x_dim,
                                            is_minimize=is_minimize)
        x0 = lagrangian.get_x0(x0)
        loop_tracker.append(f_val=lagrangian.f_inner(x0), location=x0)
        override_success = False
        for iter_outer in range(max_iter_outer_lagrangian):
            inner_res = self.unconstrained_minimize(lagrangian, x0=loop_tracker.location_values[-1], obj_tol=obj_tol,
                                                    param_tol=param_tol, max_iter=max_iter,
                                                    verbose=False, print_every=np.inf,
                                                    failure_p_step_is_success=True)
            if not inner_res['success']:
                break
            inner_location_values = inner_res['location'].reshape(-1, 1)
            assert lagrangian.eq_total_norm(inner_location_values) < param_tol
            outer_f_val = lagrangian.f_inner(inner_location_values)
            if not loop_tracker.append(f_val=outer_f_val, location=inner_location_values):
                break
            if lagrangian.obj_tol() < obj_tol:
                override_success = True
                break
            lagrangian.t *= mu_lagrangian

        self.objective_values, self.location_values, self.res = loop_tracker.summary()
        if override_success:
            self.res['success'] = True
        final_location = self.res['location'].reshape(-1, 1)
        self.res['Equality constraints values'] = lagrangian.eq_constrains_values(final_location)
        self.res['Equality constraints: Total Norm (Ax-b)'] = lagrangian.eq_total_norm(final_location)
        self.res['Inequality constraints values'] = lagrangian.ineq_constrains_values(final_location)
        print(self.res)
        self.save_function = lagrangian
        return self.res

    def unconstrained_minimize(self, f, x0, step_len=None, obj_tol=None, param_tol=None, max_iter=None, newton=None,
                               wolfe_c1=None, wolfe_backtracking=None, verbose=None, print_every=None,
                               failure_p_step_is_success=False):
        self.set_run_params(newton, step_len, wolfe_c1, wolfe_backtracking)
        if obj_tol is None:
            obj_tol = self.obj_tol
        if param_tol is None:
            param_tol = self.param_tol
        if max_iter is None:
            max_iter = self.max_iter
        if verbose is None:
            verbose = self.verbose
        if print_every is None:
            print_every = self.print_every
        loop_tracker = MinimizerLoopTracker(obj_tol=obj_tol, param_tol=param_tol, verbose=verbose,
                                            print_every=print_every,
                                            max_iter=max_iter)
        x0 = np.array(x0, dtype=float).copy().reshape(-1, 1)
        for itr in range(max_iter):
            f_res = self.apply_func_with_wolfe(f, x=x0)
            if f_res is None:
                break
            f_val = f_res[0]
            if not loop_tracker.append(f_val, x0):
                break
            p_step = f_res[3]
            if p_step is None:
                # failure
                if verbose:
                    print("Optimization failed due to invalid p_step.")
                loop_tracker.success = failure_p_step_is_success
                break
            x0 += p_step
        self.objective_values, self.location_values, res = loop_tracker.summary()
        self.save_function = f
        return res

    def apply_func_with_wolfe(self, f, x):
        newton = self.run_param['newton']
        f_res = f(x, calc_hessian=newton)
        if f_res is None:
            return None
        g_val = f_res[1]
        if g_val is None:
            return f_res + (None,)
        if newton:
            try:
                h_val = f_res[2]
                cond = np.linalg.cond(h_val)
                if not np.isfinite(cond) or cond > LineSearch.max_cond:
                    raise ValueError('Invalid condition number for Hessian')
                p_step = np.linalg.solve(h_val, -g_val)  # might throw exceptions
            except (ValueError, np.linalg.LinAlgError):
                # Hessian is not invertible --> could not converge
                return f_res + (None,)
        else:
            step_len = self.run_param['step_len']
            p_step = -step_len * g_val

        wolfe_c1 = self.run_param['wolfe_c1']
        if wolfe_c1 > 0:
            wolfe_backtracking = self.run_param['wolfe_backtracking']
            a = 1
            wolfe_number = wolfe_c1 * g_val.T.dot(p_step)
            if wolfe_number <= -LineSearch.eps:
                while a >= LineSearch.eps:
                    next_f = f(x=x + a * p_step, calc_gradient=False)
                    if next_f > f_res[0] + a * wolfe_number:
                        # still not satisfying Wolfe condition...
                        a *= wolfe_backtracking
                        continue
                    # found step size that satisfied Wolfe condition
                    p_step *= a
                    break
        return f_res + (p_step,)

    def set_run_params(self, newton=None, step_len=None, wolfe_c1=None, wolfe_backtracking=None, force=False):
        if not force and self.run_param:
            return
        if newton is None:
            newton = self.newton
        if step_len is None:
            step_len = self.step_len
        if wolfe_c1 is None:
            wolfe_c1 = self.wolfe_c1
        if wolfe_backtracking is None:
            wolfe_backtracking = self.wolfe_backtracking
        self.run_param = {'newton': newton, 'step_len': step_len, 'wolfe_c1': wolfe_c1,
                          'wolfe_backtracking': wolfe_backtracking}

    def feasible_points_values(self, points):
        if isinstance(self.save_function, Lagrangian):
            z = self.save_function.f(points, calc_gradient=False).reshape(-1)
            # eps > 0 for including the points on the feasible edges
            feasible_points = self.save_function.feasible_points(points, eps=1e-10)
            z[~feasible_points] = np.nan
        else:
            z = self.save_function(points, calc_gradient=False).reshape(-1)
        return z
