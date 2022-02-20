import os
import typing

from sklearn.gaussian_process.kernels import *
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from matplotlib import cm

# own libraries
# from sklearn.kernel_approximation import Nystroem
from time import time

# Set `EXTENDED_EVALUATION` to `True` in order to visualize your predictions.
EXTENDED_EVALUATION = True
EVALUATION_GRID_POINTS = 300  # Number of grid points used in extended evaluation
EVALUATION_GRID_POINTS_3D = 50  # Number of points displayed in 3D during evaluation




# Cost function constants
THRESHOLD = 35.5
COST_W_NORMAL = 1.0
COST_W_OVERPREDICT = 5.0
COST_W_THRESHOLD = 20.0


class Model(object):
    """
    Model for this task.
    You need to implement the fit_model and predict methods
    without changing their signatures, but are allowed to create additional methods.
    """

    def __init__(self):
        """
        Initialize your model here.
        We already provide a random number generator for reproducibility.
        """
        self.rng = np.random.default_rng(seed=0)
        self.random_int = self.rng.integers(low=0, high=50, size=1)[0]

        #kernel = WhiteKernel(noise_level=1e-02,noise_level_bounds="fixed") + RBF(length_scale=0.0246, length_scale_bounds="fixed")
        # kernel = WhiteKernel() + RBF()
        # kernel = WhiteKernel() + RationalQuadratic()
        self.gpr = GaussianProcessRegressor(kernel=WhiteKernel()+RBF(),n_restarts_optimizer=0,normalize_y=True,random_state=self.random_int)
        # self.gpr = GaussianProcessRegressor(kernel=WhiteKernel()+RationalQuadratic(),n_restarts_optimizer=0,normalize_y=True,random_state=self.random_int)
        

    def predict(self, x: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict the pollution concentration for a given set of locations.
        :param x: Locations as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :return:
            Tuple of three 1d NumPy float arrays, each of shape (NUM_SAMPLES,),
            containing your predictions, the GP posterior mean, and the GP posterior stddev (in that order)
        """


        y = self.gpr.predict(x, return_std=True)

        gp_mean = np.zeros(x.shape[0], dtype=float)
        gp_std = np.zeros(x.shape[0], dtype=float)
        gp_mean = y[0]
        gp_std = y[1]
        
        
        predictions = np.zeros(x.shape[0], dtype=float)
        for i in range(len(gp_mean)):
            if gp_mean[i] < THRESHOLD and gp_mean[i]+gp_std[i] >= THRESHOLD:
                predictions[i] = THRESHOLD
            elif gp_mean[i]-0.5*gp_std[i]  >= THRESHOLD or gp_mean[i]+gp_std[i] < THRESHOLD:
                predictions[i] = gp_mean[i] - 0.5*gp_std[i]
            else:
                predictions[i] = gp_mean[i]

        # predictions = np.zeros(x.shape[0], dtype=float)
        # for i in range(x.shape[0]):
        #    if gp_mean[i]<THRESHOLD and gp_mean[i]+gp_std[i] >= THRESHOLD:
        #        predictions[i] = THRESHOLD
        #    else:
        #        predictions[i] = gp_mean[i]

        # predictions = gp_mean

        return predictions, gp_mean, gp_std

    def fit_model(self, train_x: np.ndarray, train_y: np.ndarray):
        """
        Fit your model on the given training data.
        :param train_x: Training features as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :param train_y: Training pollution concentrations as a 1d NumPy float array of shape (NUM_SAMPLES,)
        """
      
        # t0 = time()
        # print('starting')
        
        # feature_map_nystroem = Nystroem(gamma=0.2, random_state=1, n_components=150)
        # train_x_transformed = feature_map_nystroem.fit_transform(train_x, train_y)
        # t1 = time() - t0
        # print("done in %0.3fs" % t1)
        #
        # self.gpr.fit(train_x[:3000], train_y[:3000])
        # print("done in %0.3fs" % (time() - t0))
        
        best_cost = 1000000.0
        kf = KFold(n_splits=20)
        for test_indexes, train_indexes in kf.split(train_x):
            train_x_fold, test_x_fold = train_x[train_indexes], train_x[test_indexes]
            train_y_fold, test_y_fold = train_y[train_indexes], train_y[test_indexes]
            kernel = WhiteKernel() + RBF()
            # kernel = WhiteKernel() + RationalQuadratic()
            gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2,normalize_y=True)
            gpr.fit(train_x_fold, train_y_fold)
            cost = cost_function(test_y_fold, gpr.predict(test_x_fold, return_std=True)[0])
            if cost < best_cost:
                best_cost = cost
                self.gpr = GaussianProcessRegressor(kernel=gpr.kernel_.set_params(k1__noise_level_bounds="fixed", k2__length_scale_bounds="fixed"),n_restarts_optimizer=0,normalize_y=True,random_state=self.random_int)
                # self.gpr = GaussianProcessRegressor(kernel=gpr.kernel_.set_params(k1__noise_level_bounds="fixed", k2__length_scale_bounds="fixed", k2__alpha_bounds="fixed"),n_restarts_optimizer=0,normalize_y=True,random_state=self.random_int)
                # self.gpr.kernel_ = gpr.kernel_

        
        self.gpr.fit(train_x, train_y)
        
        # print("done in %0.3fs" % (time() - t0))



def cost_function(y_true: np.ndarray, y_predicted: np.ndarray) -> float:
    """
    Calculates the cost of a set of predictions.

    :param y_true: Ground truth pollution levels as a 1d NumPy float array
    :param y_predicted: Predicted pollution levels as a 1d NumPy float array
    :return: Total cost of all predictions as a single float
    """
    assert y_true.ndim == 1 and y_predicted.ndim == 1 and y_true.shape == y_predicted.shape

    # Unweighted cost
    cost = (y_true - y_predicted) ** 2
    weights = np.zeros_like(cost)

    # Case i): overprediction
    mask_1 = y_predicted > y_true
    weights[mask_1] = COST_W_OVERPREDICT

    # Case ii): true is above threshold, prediction below
    mask_2 = (y_true >= THRESHOLD) & (y_predicted < THRESHOLD)
    weights[mask_2] = COST_W_THRESHOLD

    # Case iii): everything else
    mask_3 = ~(mask_1 | mask_2)
    weights[mask_3] = COST_W_NORMAL

    # Weigh the cost and return the average
    return np.mean(cost * weights)


def perform_extended_evaluation(model: Model, output_dir: str = '/results'):
    """
    Visualizes the predictions of a fitted model.
    :param model: Fitted model to be visualized
    :param output_dir: Directory in which the visualizations will be stored
    """
    print('Performing extended evaluation')
    fig = plt.figure(figsize=(30, 10))
    fig.suptitle('Extended visualization of task 1')

    # Visualize on a uniform grid over the entire coordinate system
    grid_lat, grid_lon = np.meshgrid(
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
    )
    visualization_xs = np.stack((grid_lon.flatten(), grid_lat.flatten()), axis=1)

    # Obtain predictions, means, and stddevs over the entire map
    predictions, gp_mean, gp_stddev = model.predict(visualization_xs)
    predictions = np.reshape(predictions, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))
    gp_mean = np.reshape(gp_mean, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))
    gp_stddev = np.reshape(gp_stddev, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))

    vmin, vmax = 0.0, 65.0
    vmax_stddev = 35.5

    # Plot the actual predictions
    ax_predictions = fig.add_subplot(1, 3, 1)
    predictions_plot = ax_predictions.imshow(predictions, vmin=vmin, vmax=vmax)
    ax_predictions.set_title('Predictions')
    fig.colorbar(predictions_plot)

    # Plot the raw GP predictions with their stddeviations
    ax_gp = fig.add_subplot(1, 3, 2, projection='3d')
    ax_gp.plot_surface(
        X=grid_lon,
        Y=grid_lat,
        Z=gp_mean,
        facecolors=cm.get_cmap()(gp_stddev / vmax_stddev),
        rcount=EVALUATION_GRID_POINTS_3D,
        ccount=EVALUATION_GRID_POINTS_3D,
        linewidth=0,
        antialiased=False
    )
    ax_gp.set_zlim(vmin, vmax)
    ax_gp.set_title('GP means, colors are GP stddev')

    # Plot the standard deviations
    ax_stddev = fig.add_subplot(1, 3, 3)
    stddev_plot = ax_stddev.imshow(gp_stddev, vmin=vmin, vmax=vmax_stddev)
    ax_stddev.set_title('GP estimated stddev')
    fig.colorbar(stddev_plot)

    # Save figure to pdf
    figure_path = os.path.join(output_dir, 'extended_evaluation.pdf')
    fig.savefig(figure_path)
    print(f'Saved extended evaluation to {figure_path}')

    plt.show()


def main():
    # Load the training dateset and test features
    train_x = np.loadtxt('train_x.csv', delimiter=',', skiprows=1)
    train_y = np.loadtxt('train_y.csv', delimiter=',', skiprows=1)
    test_x = np.loadtxt('test_x.csv', delimiter=',', skiprows=1)

    # Fit the model
    print('Fitting model')
    model = Model()
    model.fit_model(train_x, train_y)

    # Predict on the test features
    print('Predicting on test features')
    predicted_y = model.predict(test_x)
    print(predicted_y)

    if EXTENDED_EVALUATION:
        perform_extended_evaluation(model, output_dir='.')


if __name__ == "__main__":
    main()
