import numpy as np
import pandas as pd
import random

import GPy
from sklearn.multioutput import MultiOutputRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
from sklearn.gaussian_process import GaussianProcessRegressor

class GP:

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    """## Function Setting (Dynamics with GP Prediction)"""

    # Define a Gaussian process

    def skl_GP(self, Xtrain, Ytrain):
        # fit Gaussian Process with dataset X_train, Y_train

        # kernel = RBF()  # define a kernel function here #
        kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-15, 1e15), nu=1.5) + WhiteKernel(noise_level_bounds=(1e-15, 1e15))
        # kernel = RationalQuadratic()

        n_restarts_optimizer = 9  # define a n_restarts_optimizerint value here #
        gp_ = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=n_restarts_optimizer, random_state=123)
        gp_.fit(Xtrain, Ytrain)
        return gp_

    # prediction function
    def skl_fit(self):
        gp1 = self.skl_GP(self.X, self.Y[:, 0].reshape(-1, 1))
        gp2 = self.skl_GP(self.X, self.Y[:, 1].reshape(-1, 1))
        self.gp1 = gp1
        self.gp2 = gp2
    
    def skl_learned_f(self, X):
        X = np.array(X).reshape(1, -1)
        y1, s1 = self.gp1.predict(X, return_std=True)
        y2, s2 = self.gp2.predict(X, return_std=True)
        return np.concatenate((y1, y2), axis=1), np.concatenate((s1, s2), axis=0).reshape(1, -1)

    def skl_f_mean(self,X):
        y1 = self.gp1.predict([X])[0][0]
        y2 = self.gp2.predict([X])[0][0]
        return [y1, y2]

    # GPy
    def gpy_GP(self, X_train, Y_train):
        dim = X_train.shape[1]
        kern = GPy.kern.Matern32(input_dim=dim, ARD=False)
        gp = GPy.models.GPRegression(X_train, Y_train.reshape(-1, 1), kern)
        return gp

    def gpy_fit(self):
        gp1 = self.gpy_GP(self.X, self.Y[:, 0].reshape(-1, 1))
        gp1.optimize()
        gp2 = self.gpy_GP(self.X, self.Y[:, 1].reshape(-1, 1))
        gp2.optimize()
        self.gp1 = gp1
        self.gp2 = gp2

    def gpy_learned_f(self, X):
        X = np.array(X).reshape(1, -1)
        y1, s1 = self.gp1.predict(X)
        y2, s2 = self.gp2.predict(X)
        return np.concatenate((y1, y2), axis=1), np.concatenate((s1, s2), axis=0).reshape(1, -1)

    def gpy_mean(self, X):
        return self.gpy_learned_f(X)[0][0]

    def J(self, X):
        grad_1, s1 = self.gp1.predictive_gradients(np.array(X).reshape(-1, 2))
        grad_2, s2 = self.gp2.predictive_gradients(np.array(X).reshape(-1, 2))
        return np.concatenate((grad_1[0].T, grad_2[0].T), axis=0), np.concatenate((s1, s2), axis=1)


