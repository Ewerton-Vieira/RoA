# import Pendulum_lc as Pd

import CMGDB_util
import CMGDB
import RoA
import dyn_tools
import Grid
import sys

import TimeMap

import numpy as np

from sklearn.multioutput import MultiOutputRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, ExpSineSquared, ConstantKernel


import matplotlib.pyplot as plt

from datetime import datetime


if __name__ == "__main__":
    MG_util = CMGDB_util.CMGDB_util()

    sb = 14
    time = 1  # time is equal to 10s

    # subdiv_min = 10  # minimal subdivision to compute Morse Graph
    # subdiv_max = 10  # maximal subdivision to compute Morse Graph
    subdiv_init = subdiv_min = subdiv_max = sb  # non adaptive proceedure

    x_min = -3.14159
    x_max = 3.14159

    y_min = -6.28318
    y_max = 6.28318

    # base name for the output files.
    base_name = "pendulum_lqr_GP_time" + \
        str(time) + "_" + \
        str(subdiv_init)

    print(base_name)

    # Define the parameters for CMGDB
    lower_bounds = [x_min, y_min]
    upper_bounds = [x_max, y_max]

    # load map
    # set the time step
    TM = TimeMap.TimeMap("pendulum_lc", time,
                         "examples/tripods/lc_roa.yaml")

    # define the lqr time map for the pendulum

    def g(X):
        Y = TM.pendulum_lqr(X)
        return Y

    phase_periodic = [True, False]

    def sample_points(lower_bounds, upper_bounds, num_pts):
        # Sample num_pts in dimension dim, where each
        # component of the sampled points are in the
        # ranges given by lower_bounds and upper_bounds
        dim = len(lower_bounds)
        X = np.random.uniform(lower_bounds, upper_bounds, size=(num_pts, dim))
        return X

    # Define a Gaussian process

    def GP(X_train, Y_train):
        # fit Gaussian Process with dataset X_train, Y_train

        # DO #
        kernel = RBF()  # define a kernel function here #

        # DO #
        n_restarts_optimizer = 9  # define a n_restarts_optimizerint value here #

        gp_ = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=n_restarts_optimizer)

        # fit multi-response independently
        # multi_reg = MultiOutputRegressor(gp)
        # multi_reg.fit(X_train, Y_train)
        # return multi_reg

        gp_.fit(X_train, Y_train)
        return gp_

    np.random.seed(123)  # specify a seed to generate points #

    # DO #
    n = 500  # specify a number of initial points #

    # generate training data
    X = sample_points(lower_bounds, upper_bounds, n)
    Y = [g(x_) for x_ in X]
    Y = np.array(Y)

    # train GP regression with X and Y
    gp = GP(X, Y)

    # prediction function

    def learned_f(X):
        return gp.predict(X, return_std=True)

    print(learned_f([[1, 1]]))
    K = 1

    ############

    def F(rect):
        return MG_util.Box_GP_K(learned_f, rect, K)
        # return CMGDB.BoxMap(g, rect, padding=True)
        # return MG_util.F_K(g, rect, K)
        # return MG_util.BoxMapK(g_on_grid, rect, K)

    morse_graph, map_graph = MG_util.run_CMGDB(
        subdiv_min, subdiv_max, lower_bounds, upper_bounds, phase_periodic, F, base_name, subdiv_init)

    # CMGDB.PlotMorseSets(morse_graph)

    startTime = datetime.now()

    roa = RoA.RoA(map_graph, morse_graph)

    print(f"Time to build the regions of attraction = {datetime.now() - startTime}")

    # roa.save_file(base_name)

    fig, ax = roa.PlotTiles()

    # RoA.PlotTiles(lower_bounds, upper_bounds,
    #               from_file=base_name, from_file_basic=True)

    # plt.show()

    # roa.save_file(base_name)

    ########

    def F(rect):
        return MG_util.F_GP_K(learned_f, rect, K)
        # return CMGDB.BoxMap(g, rect, padding=True)
        # return MG_util.F_K(g, rect, K)
        # return MG_util.BoxMapK(g_on_grid, rect, K)

    morse_graph, map_graph = MG_util.run_CMGDB(
        subdiv_min, subdiv_max, lower_bounds, upper_bounds, phase_periodic, F, base_name, subdiv_init)

    # CMGDB.PlotMorseSets(morse_graph)

    startTime = datetime.now()

    roa = RoA.RoA(map_graph, morse_graph)

    print(f"Time to build the regions of attraction = {datetime.now() - startTime}")

    # roa.save_file(base_name)

    fig, ax = roa.PlotTiles()

    # RoA.PlotTiles(lower_bounds, upper_bounds,
    #               from_file=base_name, from_file_basic=True)

    # plt.show()

    # roa.save_file(base_name)

    ##########

    def g(X):
        return gp.predict([X])[0]

    def F(rect):
        # return MG_util.Box_GP_K(learned_f, rect, K)
        return CMGDB.BoxMap(g, rect, padding=True)
        # return MG_util.F_K(g, rect, K)
        # return MG_util.BoxMapK(g_on_grid, rect, K)

    morse_graph, map_graph = MG_util.run_CMGDB(
        subdiv_min, subdiv_max, lower_bounds, upper_bounds, phase_periodic, F, base_name, subdiv_init)

    # CMGDB.PlotMorseSets(morse_graph)

    startTime = datetime.now()

    roa = RoA.RoA(map_graph, morse_graph)

    print(f"Time to build the regions of attraction = {datetime.now() - startTime}")

    # roa.save_file(base_name)

    fig, ax = roa.PlotTiles()

    # RoA.PlotTiles(lower_bounds, upper_bounds,
    #               from_file=base_name, from_file_basic=True)

    plt.show()

    # roa.save_file(base_name)
