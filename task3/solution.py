import random
import os
import typing
import logging
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import matplotlib.pyplot as plt

# WE IMPORTED
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *
from scipy.stats import norm

EXTENDED_EVALUATION = True
# Set `EXTENDED_EVALUATION` to `True` in order to visualize your predictions.


""" Solution """


class BO_algo(object):
    def __init__(self):
        """Initializes the algorithm with a parameter configuration. """

        # TODO: enter your code here
        self.previous_points = []

        self.rng = np.random.default_rng(seed=0)
        self.random_int = self.rng.integers(low=0, high=50, size=1)[0]
        # IMPORTANT: DO NOT REMOVE THOSE ATTRIBUTES AND USE sklearn.gaussian_process.GaussianProcessRegressor instances!
        # Otherwise, the extended evaluation will break.
        self.noise_f = 0.01
        self.noise_v = 0.005

        self.constraint_kernel = ConstantKernel(constant_value=3.5)*RBF(length_scale=2, length_scale_bounds="fixed")
        self.objective_kernel = ConstantKernel(constant_value=1.5)*RBF(length_scale=1.5, length_scale_bounds="fixed")

        self.constraint_model = GaussianProcessRegressor(kernel=self.constraint_kernel, alpha=self.noise_v**2, random_state=self.random_int)  #, normalize_y=True, random_state=self.random_int)  # alpha=self.noise_v**2)
        self.objective_model = GaussianProcessRegressor(kernel=self.objective_kernel, alpha=self.noise_f**2, random_state=self.random_int)  #, normalize_y=True, random_state=self.random_int)  # alpha=self.noise_f**2)

        # kappa for GP-LCB
        self.kappa = 5

    def next_recommendation(self) -> np.ndarray:
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: np.ndarray
            1 x domain.shape[0] array containing the next point to evaluate
        """

        # TODO: enter your code here
        # In implementing this function, you may use optimize_acquisition_function() defined below.
        if len(self.previous_points) == 0:
            # ---------A)----------------
            # x = np.array([(domain_x[0, 1] - domain_x[0, 0])/2])
            # x.resize(1, domain_x.shape[0])
            # ---------B)----------------
            # x = np.array([np.random.uniform(0, 6), np.random.uniform(0, 6)])
            # ---------C)----------------
            x1 = (domain_x[0, 1] - domain_x[0, 0]) / 2.0
            x2 = (domain_x[1, 1] - domain_x[1, 0]) / 2.0
            x = np.array([[x1, x2]])
        else:
            x = self.optimize_acquisition_function()

        return np.atleast_2d(x)

    def optimize_acquisition_function(self) -> np.ndarray:  # DON'T MODIFY THIS FUNCTION
        """
        Optimizes the acquisition function.

        Returns
        -------
        x_opt: np.ndarray
            1 x domain.shape[0] array containing the point that approximately maximizes the acquisition function.
        """

        def objective(x: np.array):
            return - self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick best solution
        for _ in range(20):
            x0 = domain_x[0, 0] + (domain_x[0, 1] - domain_x[0, 0]) * \
                 np.random.rand(1)
            x1 = domain_x[1, 0] + (domain_x[1, 1] - domain_x[1, 0]) * \
                 np.random.rand(1)
            result = fmin_l_bfgs_b(objective, x0=np.array([x0, x1]), bounds=domain_x,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *domain_x[0]))
            f_values.append(result[1])

        ind = np.argmin(f_values)
        return np.atleast_2d(x_values[ind])

    def acquisition_function(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the acquisition function.

        Parameters
        ----------
        x: np.ndarray
            point in the domain of f

        Returns
        ------
        af_value: float
            value of the acquisition function at x
        """

        # TODO: enter your code here
        f_hat, sigma_f_hat = self.objective_model.predict(np.atleast_2d(x), return_std=True)
        v_hat, sigma_v_hat = self.constraint_model.predict(np.atleast_2d(x), return_std=True)

        # ------ Calculation of gamma of Chapter 2.2 in Snoek et al. (2012)-------------
        # -----------------A)------------------------
        # all_points = np.asarray(self.previous_points)
        # while all_points.any():
        #     min_idx = np.argmin(all_points, axis=0)[2]
        #     if all_points[min_idx, 3] < -1:
        #         break
        #     all_points = np.delete(all_points, min_idx, 0)
        # if all_points.size == 0:
        #     z_best = 1000
        # else:
        #     z_best = all_points[min_idx, 2]
        # gamma = (z_best - f_hat - self.noise_f)/sigma_f_hat
        # ------------------B)---------------------------
        all_points = np.asarray(self.previous_points)
        feasible_points = all_points[all_points[:, 3] < -self.noise_v]
        if feasible_points.size == 0:
            min_idx = np.argmin(all_points[:, 3])
            z_best = all_points[min_idx, 2]
        else:
            min_idx = np.argmin(feasible_points[:, 2])
            z_best = feasible_points[min_idx, 2]
        gamma = (z_best - f_hat)/(sigma_f_hat + 1E-9)
        # ----------Probability of impovement (PI) method according to Chapter 2.2 in Snoek et al. (2012)-------------
        # af_value_unconstr = norm.cdf(gamma, loc=f_hat, scale=sigma_f_hat)

        # -----------Expected Improvement (EI) method according to Chapter 2.2 in Snoek et al. (2012)-----------------
        # af_value_unconstr = sigma_f_hat*(gamma*norm.cdf(gamma, loc=f_hat, scale=sigma_f_hat) + norm.pdf(gamma, loc=0, scale=1))
        af_value_unconstr = sigma_f_hat*(gamma*norm.cdf(gamma) + norm.pdf(gamma))

        # ------------GP-LCB according to Chapter 2.2 in Snoek et al. (2012)------------------------------------------
        # af_value_unconstr = f_hat - self.kappa*sigma_f_hat

        # -------------Calculation of Pr(C(x)) according to Chapter 3.1 in Gelbart et al. 2014-------------------------
        prob_c = norm.cdf(0, loc=v_hat, scale=sigma_v_hat)
        # prob_c = 1 - norm.cdf(0, loc=v_hat, scale=sigma_v_hat)
        # prob_c = 1 + norm.cdf(0, loc=v_hat, scale=sigma_v_hat)
        # prob_c = norm.cdf(v_hat/sigma_v_hat)
        # dist_v = norm(loc=v_hat, scale=sigma_v_hat)
        # prob_c = 1 - dist_v.cdf(1.2)

        # thresh = 0.75
        # if prob_c < thresh:
        #    prob_c = 0

        # constrained acquisition function according to equation (7) in Chapter 3.1 in Gelbart et al. 2014
        af_value = af_value_unconstr*prob_c
        return af_value

    def add_data_point(self, x: np.ndarray, z: float, c: float):
        """
        Add data points to the model.

        Parameters
        ----------
        x: np.ndarray
            point in the domain of f
        z: np.ndarray
            value of the acquisition function at x
        c: np.ndarray
            value of the condition function at x
        """

        assert x.shape == (1, 2)
        self.previous_points.append([float(x[:, 0]), float(x[:, 1]), float(z), float(c)])
        # TODO: enter your code here
        current_points = np.asarray(self.previous_points)
        self.objective_model.fit(current_points[:, :2], current_points[:, 2])
        self.constraint_model.fit(current_points[:, :2], current_points[:, 3])

    def get_solution(self) -> np.ndarray:
        """
        Return x_opt that is believed to be the minimizer of f.

        Returns
        -------
        solution: np.ndarray
            1 x domain.shape[0] array containing the optimal solution of the problem
        """

        # TODO: enter your code here
        # 3 OPTIONS
        # --------------------A)--------------------------------
        all_points = np.asarray(self.previous_points)
        min_idx = -1
        while all_points.any():
            min_idx = np.argmin(all_points, axis=0)[2]
            if all_points[min_idx, 3] < 0:
                break
            all_points = np.delete(all_points, min_idx, 0)

        sol = all_points[min_idx, :2]

        # --------------------B)--------------------------------
        # all_points = np.asarray(self.previous_points)
        # feasible_points = all_points[all_points[:, 3] < -self.noise_v]
        # if feasible_points.size == 0:
        #     min_idx = np.argmin(all_points[:, 3])
        #     sol = all_points[min_idx, :2]
        # else:
        #     min_idx = np.argmin(feasible_points[:, 2])
        #     sol = feasible_points[min_idx, :2]

        # -----------------C)-----------------------------------
        # all_points = np.asarray(self.previous_points)
        # ok_points = all_points[all_points[:, 3] < -self.noise_v]
        # data_post_f = ok_points.copy()
        # for i in range(len(ok_points)):
        #     data_post_f[i] = self.objective_model.predict(np.atleast_2d(ok_points[i, :2]))
        #
        # if len(ok_points) > 0:
        #     i = np.argmin(data_post_f[:, 2])
        #     sol = ok_points[i, :2]
        # else:
        #     i = np.argmin(all_points[:, 2])
        #     sol = all_points[i, :2]

        return sol


""" 
    Toy problem to check  you code works as expected
    IMPORTANT: This example is never used and has nothing in common with the task you
    are evaluated on, it's here only for development and illustration purposes.
"""
domain_x = np.array([[0, 6], [0, 6]])
EVALUATION_GRID_POINTS = 250
CONSTRAINT_OFFSET = - 0.8  # This is an offset you can change to make the constraint more or less difficult to fulfill
LAMBDA = 0.0  # You shouldn't change this value


def check_in_domain(x) -> bool:
    """Validate input"""
    x = np.atleast_2d(x)
    v_dim_0 = np.all(x[:, 0] >= domain_x[0, 0]) and np.all(x[:, 0] <= domain_x[0, 1])
    v_dim_1 = np.all(x[:, 1] >= domain_x[1, 0]) and np.all(x[:, 0] <= domain_x[1, 1])

    return v_dim_0 and v_dim_1


def f(x) -> np.ndarray:
    """Dummy objective"""
    l1 = lambda x0, x1: np.sin(x0) + x1 - 1

    return l1(x[:, 0], x[:, 1])


def c(x) -> np.ndarray:
    """Dummy constraint"""
    c1 = lambda x, y: np.cos(x) * np.cos(y) - 0.1

    return c1(x[:, 0], x[:, 1]) - CONSTRAINT_OFFSET


def get_valid_opt(f, c, domain) -> typing.Tuple[float, float, np.ndarray, np.ndarray]:
    nx, ny = (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS)
    x = np.linspace(domain[0, 0], domain[0, 1], nx)
    y = np.linspace(domain[1, 0], domain[1, 1], ny)
    xv, yv = np.meshgrid(x, y)
    samples = np.array([xv.reshape(-1), yv.reshape(-1)]).T

    true_values = f(samples)
    true_cond = c(samples)
    valid_data_idx = np.where(true_cond < LAMBDA)[0]
    f_opt = np.min(true_values[np.where(true_cond < LAMBDA)])
    x_opt = samples[valid_data_idx][np.argmin(true_values[np.where(true_cond < LAMBDA)])]
    f_max = np.max(np.abs(true_values))
    x_max = np.argmax(np.abs(true_values))
    return f_opt, f_max, x_opt, x_max


def perform_extended_evaluation(agent, output_dir='./'):
    fig = plt.figure(figsize=(25, 5), dpi=50)
    nx, ny = (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS)
    x = np.linspace(0.0, 6.0, nx)
    y = np.linspace(0.0, 6.0, ny)
    xv, yv = np.meshgrid(x, y)
    x_b, y_b = agent.get_solution()
    samples = np.array([xv.reshape(-1), yv.reshape(-1)]).T
    predictions, stds = agent.objective_model.predict(samples, return_std=True)
    predictions = np.reshape(predictions, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))
    conds = agent.constraint_model.predict(samples)
    conds = np.reshape(conds, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))

    true_values = f(samples)
    true_cond = c(samples)
    conditions_verif = (true_cond < LAMBDA).astype(float)
    conditions_with_nans = 1 - np.copy(conditions_verif)
    conditions_with_nans[np.where(conditions_with_nans == 0)] = np.nan
    conditions_with_nans = np.reshape(conditions_with_nans, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))
    valid_data_idx = np.where(true_cond < LAMBDA)[0]

    f_opt = np.min(true_values[np.where(true_cond < LAMBDA)])
    x_opt = samples[valid_data_idx][np.argmin(true_values[np.where(true_cond < LAMBDA)])]

    sampled_point = np.array(agent.previous_points)

    ax_condition = fig.add_subplot(1, 4, 4)
    im_cond = ax_condition.pcolormesh(xv, yv, conds.reshape((EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS)), shading='auto',
                                      linewidth=0)
    im_cond.set_edgecolor('face')
    fig.colorbar(im_cond, ax=ax_condition)
    ax_condition.scatter(sampled_point[:, 0], sampled_point[:, 1], cmap='Blues', marker='x',
                         label='Sampled Point by BO', antialiased=True, linewidth=0)
    ax_condition.pcolormesh(xv, yv, conditions_with_nans, shading='auto', cmap='Reds', alpha=0.7, vmin=0, vmax=1.0,
                            linewidth=0, antialiased=True)
    ax_condition.set_title('Constraint GP Posterior +  True Constraint (Red is Infeasible)')
    ax_condition.legend(fontsize='x-small')

    ax_gp_f = fig.add_subplot(1, 4, 2, projection='3d')
    ax_gp_f.plot_surface(
        X=xv,
        Y=yv,
        Z=predictions,
        rcount=100,
        ccount=100,
        linewidth=0,
        antialiased=False
    )
    ax_gp_f.set_title('Posterior 3D for Objective')

    ax_gp_c = fig.add_subplot(1, 4, 3, projection='3d')
    ax_gp_c.plot_surface(
        X=xv,
        Y=yv,
        Z=conds,
        rcount=100,
        ccount=100,
        linewidth=0,
        antialiased=False
    )
    ax_gp_c.set_title('Posterior 3D for Constraint')

    ax_predictions = fig.add_subplot(1, 4, 1)
    im_predictions = ax_predictions.pcolormesh(xv, yv, predictions, shading='auto', label='Posterior',linewidth=0, antialiased=True)
    im_predictions.set_edgecolor('face')
    fig.colorbar(im_predictions, ax=ax_predictions)
    ax_predictions.pcolormesh(xv, yv, conditions_with_nans, shading='auto', cmap='Reds', alpha=0.7, vmin=0, vmax=1.0,
                              label=' True Infeasible',linewidth=0, antialiased=True)
    ax_predictions.scatter(x_b, y_b, s=20, marker='x', label='Predicted Value by BO')
    ax_predictions.scatter(x_opt[0], x_opt[1], s=20, marker='o', label='True Optimimum Under Constraint')
    ax_predictions.set_title('Objective GP Posterior + True Constraint (Red is Infeasible)')
    ax_predictions.legend(fontsize='x-small')
    figure_path = os.path.join(output_dir, 'extended_evaluation.pdf')
    fig.savefig(figure_path)
    plt.show()


def train_on_toy(agent, iteration):
    logging.info('Running model on toy example.')
    seed = 1234
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    for j in range(iteration):
        # Get next recommendation
        x = agent.next_recommendation()

        # Check for valid shape
        assert x.shape == (1, domain_x.shape[0]), \
            f"The function next recommendation must return a numpy array of " \
            f"shape (1, {domain_x.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x) + np.random.normal(size=(x.shape[0],), scale=0.01)
        cost_val = c(x) + np.random.normal(size=(x.shape[0],), scale=0.005)
        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = np.atleast_2d(agent.get_solution())
    assert solution.shape == (1, domain_x.shape[0]), \
        f"The function get solution must return a numpy array of shape (" \
        f"1, {domain_x.shape[0]})"

    assert check_in_domain(solution), \
        f'The function get solution must return a point within the ' \
        f'domain, {solution} returned instead'

    # Compute regret
    f_opt, f_max, x_opt, x_max = get_valid_opt(f, c, domain_x)
    if c(solution) > 0.0:
        regret = 1
    else:
        regret = (f(solution) - f_opt) / f_max

    print(f'Optimal value: {f_opt}\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret{regret}')
    return agent


def main():
    logging.warning(
        'This main method is for illustrative purposes only and will NEVER be called by the checker!\n'
        'The checker always calls run_solution directly.\n'
        'Please implement your solution exclusively in the methods and classes mentioned in the task description.'
    )

    seed = 1234
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    agent = BO_algo()

    agent = train_on_toy(agent, 20)

    if EXTENDED_EVALUATION:
        perform_extended_evaluation(agent)


if __name__ == "__main__":
    main()
