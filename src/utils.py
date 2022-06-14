import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def plot_line_search(func_name, line_search_results):
    fig, ax = plt.subplots(figsize=(7, 7))
    plt.title("Convergence of function: " + func_name)
    if not isinstance(line_search_results, list):
        line_search_results = [line_search_results]
    colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(line_search_results)))
    max_iter = max([len(ls) for ls in line_search_results])
    min_iter = min([len(ls) for ls in line_search_results])
    use_log_x = max_iter > 10 * min_iter
    # resort such that shorter path will appear above longer path
    line_search_results = sorted(line_search_results, key=lambda ls_res: len(ls_res), reverse=True)
    for ind, ls in enumerate(line_search_results):
        objective_values = ls.objective_values
        num_vals = len(objective_values)
        color = colors[ind]
        label = get_plot_label(ls)
        a_range = np.arange(1, num_vals + 1) if use_log_x else np.arange(num_vals)
        if num_vals <= 2:
            plt.scatter(a_range, objective_values, label=label, linewidth=2, color=color)
        else:
            plt.plot(a_range, objective_values, label=label, color=color)
    min_y = min([ls.res['min_objective_value'] for ls in line_search_results])
    max_y = max([ls.res['max_objective_value'] for ls in line_search_results])
    if min_y < 0:
        ax.set_ylim(top=max_y + 0.2 * (max_y - min_y))
    ax.set_ylabel("Objective function value")
    ax.set_xlabel("Number of iterations")
    if use_log_x:
        ax.set_xlim(left=0.9, right=max_iter + 0.1)
        if max_iter > 10 * min_iter:
            ax.set_xscale('log')
            formatter = matplotlib.ticker.FuncFormatter(lambda y_format, _: '{:.16g}'.format(y_format))
            ax.xaxis.set_major_formatter(formatter)
    plt.legend(loc="upper right", prop={'size': 6})
    plt.show()
    max_norm = max([ls.res['max_norm_location'] for ls in line_search_results])
    data_dim = line_search_results[0].res['dim']
    if data_dim == 2:
        # 3D plot
        regions_contours = [np.linspace(-max_norm, max_norm, 50)] * data_dim
        x_matrix, y_matrix = np.meshgrid(*regions_contours)
        points = np.vstack([x_matrix.reshape(1, -1), y_matrix.reshape(1, -1)])
        z_matrix = line_search_results[0].feasible_points_values(points=points).reshape(x_matrix.shape)
        fig = plt.figure(figsize=(12, 7))
        ax = fig.add_subplot(projection='3d')
        surf = ax.plot_surface(x_matrix, y_matrix, z_matrix, cmap=matplotlib.cm.gist_heat_r, linewidth=0,
                               antialiased=False, alpha=0.5)
        min_z, max_z = np.nanmin(z_matrix), np.nanmax(z_matrix)
        min_z, max_z = min(min_z, 0), max(max_z, 0)
        eps_z = 0.05 * (max_z - min_z)
        ax.set_zlim(min_z, max_z)
        fig.colorbar(surf, shrink=0.5, aspect=10)
        plt.title("3D plot of " + func_name)
        for ind, ls in enumerate(line_search_results):
            objective_values = ls.objective_values + eps_z
            color = colors[ind]
            xs, ys = ls.location_values[0], ls.location_values[1]
            label = get_plot_label(ls)
            if len(objective_values) <= 2:
                ax.scatter(xs, ys, objective_values, label=label, linewidth=3, color=color)
            else:
                ax.plot(xs, ys, objective_values, label=label, linewidth=2, color=color)
        ax.set_xlabel("X values")
        ax.set_ylabel("Y values")
        ax.set_zlabel("Objective function value")
        plt.legend()
        plt.show()
        # plot contours
        plt.figure(figsize=(12, 7))
        plt.contour(x_matrix, y_matrix, z_matrix, 200)
        plt.title("2D contours plot of " + func_name)
        for ind, ls in enumerate(line_search_results):
            objective_values = ls.objective_values
            color = colors[ind]
            xs, ys = ls.location_values[0], ls.location_values[1]
            label = get_plot_label(ls)
            if len(objective_values) <= 2:
                plt.scatter(xs, ys, label=label, linewidth=3, color=color)
            else:
                plt.plot(xs, ys, label=label, linewidth=2, color=color)
    else:
        fig = plt.figure()
        ax = Axes3D(fig, auto_add_to_figure=False)
        fig.add_axes(ax)
        ax.set_title("3D plot of " + func_name)
        for ind, ls in enumerate(line_search_results):
            color = colors[ind]
            ineqs = -np.hstack([ineq.q1.reshape(-1, 1) for ineq in ls.save_function.inequalities])
            ineqs = [[tuple(row) for row in ineqs]]
            poly_3d_collection = Poly3DCollection(ineqs, alpha=0.7, edgecolors="r")
            ax.add_collection3d(poly_3d_collection)
            xs = ls.location_values[0]
            ys = ls.location_values[1]
            zs = ls.location_values[2]
            label = get_plot_label(ls)
            ax.plot(xs, ys, zs, label=label, color=color, marker=".", linestyle="--")
        ax.set_zlabel("Z values")

    plt.xlabel("X values")
    plt.ylabel("Y values")
    plt.legend()
    plt.show()


def get_plot_label(ls):
    run_param = ls.run_param
    ret = "Newton Step Method" if run_param['newton'] else "Gradient Descent (step_len={})".format(
        run_param['step_len'])
    if run_param['wolfe_c1'] > 0:
        ret += " Wolfe:[{},{}]".format(run_param['wolfe_c1'], run_param['wolfe_backtracking'])
    return ret
