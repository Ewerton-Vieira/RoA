# CMGDB_util.py  # 2021-10-26
# MIT LICENSE 2020 Ewerton R. Vieira


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
import csv
import time
import os
import CMGDB
from datetime import datetime
import time


class CMGDB_util:

    def run_CMGDB(self, subdiv_min, subdiv_max, lower_bounds, upper_bounds, phase_periodic, F, base_name, subdiv_init=6, subdiv_limit=10000):
        # Define the parameters for CMGDB

        model = CMGDB.Model(subdiv_min, subdiv_max, subdiv_init, subdiv_limit,
                            lower_bounds, upper_bounds, phase_periodic, F)

        startTime = datetime.now()

        morse_graph, map_graph = CMGDB.ComputeMorseGraph(model)

        print(datetime.now() - startTime)

        # Save Morse graph
        dir_path = os.path.abspath(os.getcwd()) + "/output/"
        MG = dir_path + base_name
        morse_fname = dir_path + base_name + ".csv"

        CMGDB.PlotMorseGraph(morse_graph).format = 'png'
        CMGDB.PlotMorseGraph(morse_graph).render(MG)

        # Save file
        morse_nodes = range(morse_graph.num_vertices())
        morse_sets = [box + [node]
                      for node in morse_nodes for box in morse_graph.morse_set_boxes(node)]
        np.savetxt(morse_fname, np.array(morse_sets), delimiter=',')

        return morse_graph, map_graph

    # def run_CMGDB(phase_subdiv, lower_bounds, upper_bounds, phase_periodic, F, base_name):
    #     # Define the parameters for CMGDB
    #
    #     model = CMGDB.Model(phase_subdiv, lower_bounds, upper_bounds, phase_periodic, F)
    #
    #     startTime = datetime.now()
    #
    #     morse_graph, map_graph = CMGDB.ComputeMorseGraph(model)
    #
    #     print(datetime.now() - startTime)
    #
    #     # Save Morse graph
    #     dir_path = os.path.abspath(os.getcwd()) + "/output/"
    #     MG = dir_path + base_name
    #     morse_fname = dir_path + base_name + ".csv"
    #
    #     CMGDB.PlotMorseGraph(morse_graph).format = 'png'
    #     CMGDB.PlotMorseGraph(morse_graph).render(MG)
    #
    #     # Save file
    #     morse_nodes = range(morse_graph.num_vertices())
    #     morse_sets = [box + [node] for node in morse_nodes for box in morse_graph.morse_set_boxes(node)]
    #     np.savetxt(morse_fname, np.array(morse_sets), delimiter=',')
    #
    #     return morse_graph, map_graph

    def F_K(self, f, rect, K):
        """Input: function f, rectangle rect, and the Lipschit constant in vector form K
        Output: Image of rectangle by f, also taking in account the expansion given by K"""
        half = len(rect) // 2
        im_center = f([rect[i] + (rect[half + i] - rect[i])/2 for i in range(half)])
        list1 = []
        list2 = []
        for i in range(half):  # image of the center of the rect +or- lenght * K
            list1.append(im_center[i] - (rect[half + i] - rect[i]) * K[i] / 2)
            list2.append(im_center[i] + (rect[half + i] - rect[i]) * K[i] / 2)
        return list1 + list2

    def Morse_sets_vol(self, name_file):
        """Compute the volume of a Morse set"""
        with open(name_file, 'r') as f:
            lines = csv.reader(f, delimiter=',')
            d_vol = dict()
            for row in lines:
                size = len(row) - 1
                half = int(size / 2)
                volume_cube = 1
                for i in range(half):
                    volume_cube *= float(row[half + i]) - float(row[i])
                if row[size] in d_vol.keys():
                    d_vol[row[size]] = d_vol[row[size]] + volume_cube
                else:
                    d_vol[row[size]] = volume_cube
        return d_vol

    def sample_points(self, lower_bounds, upper_bounds, num_pts):
        # Sample num_pts in dimension dim, where each
        # component of the sampled points are in the
        # ranges given by lower_bounds and upper_bounds
        dim = len(lower_bounds)
        X = np.random.uniform(lower_bounds, upper_bounds, size=(num_pts, dim))
        return X

    def BoxMapK(self, f, rect, K):
        dim = int(len(rect) / 2)
        X = CMGDB.CornerPoints(rect)
        # Evaluate f at point in X
        Y = [f(x) for x in X]
        # Get lower and upper bounds of Y
        Y_l_bounds = [min([y[d] for y in Y]) - K*(rect[d + dim] - rect[d]) for d in range(dim)]
        Y_u_bounds = [max([y[d] for y in Y]) + K*(rect[d + dim] - rect[d]) for d in range(dim)]
        return Y_l_bounds + Y_u_bounds

    def Box_GP_K(self, learned_f, rect, K, n=-3):
        """learned_f with predicted mean and standard deviation
        K Lipschit constant"""
        dim = int(len(rect) / 2)
        X = CMGDB.CornerPoints(rect)
        # print(X)
        # Evaluate f at point in X

        Y, S = learned_f(X[0])

        X = X[1::]
        for x_ in X:
            y_, s_ = learned_f(x_)
            Y = np.concatenate((Y, y_))
            S = np.concatenate((S, s_))

        # print(f"{Y}\n {S} \n {S_}")
        Y_max = Y + S*(2**n)
        Y_min = Y - S*(2**n)

        # print(f"{Y_min}\n {Y_max}")

        # print(f"{np.amin(Y_min[:,0])}\n {np.amax(Y_max[:,1])}")

        # Get lower and upper bounds of Y
        Y_l_bounds = [np.amin(Y_min[:, d]) - K*(rect[d + dim] - rect[d]) for d in range(dim)]
        Y_u_bounds = [np.amax(Y_max[:, d]) + K*(rect[d + dim] - rect[d]) for d in range(dim)]
        return Y_l_bounds + Y_u_bounds

    def F_GP_K(self, learned_f, rect, K, n=-3):
        """learned_f with predicted mean and standard deviation
        K Lipschit constant"""
        dim = int(len(rect) / 2)
        X = CMGDB.CenterPoint(rect)
        # print(X)
        # Evaluate f at point in X
        Y, S = learned_f(X)

        # print(f"{Y}\n {S} \n {S_}")
        Y_max = Y + S*(2**n)
        Y_min = Y - S*(2**n)

        # print(f"{Y_min}\n {Y_max}")

        # print(f"{np.amin(Y_min[:,0])}\n {np.amax(Y_max[:,1])}")

        # Get lower and upper bounds of Y
        Y_l_bounds = [np.amin(Y_min[:, d]) - K*(rect[d + dim] - rect[d]) for d in range(dim)]
        Y_u_bounds = [np.amax(Y_max[:, d]) + K*(rect[d + dim] - rect[d]) for d in range(dim)]
        return Y_l_bounds + Y_u_bounds

    def Box_J(self, f, J, rect):
        """f: function, J: Jacobian matrix, rect: rectangle
        Given a rect return the smallest rectangle that contains the image of the
        J \cdot rect"""
        dim = int(len(rect) / 2)
        x = rect[0:dim]
        y = f(x)
        Jac, _ = J(np.array(x).reshape(-1, dim))
        # next, add the sum of the columns
        Jac = np.concatenate((Jac, Jac.sum(axis=0).reshape(1, dim)), axis=0)

        Y_l_bounds = []
        Y_u_bounds = []
        for d in range(dim):
            Y_l_bounds.append(
                np.amin(Jac[:, d]) * (rect[d + dim] - rect[d]) + y[d]
            )

            Y_u_bounds.append(
                np.amax(Jac[:, d]) * (rect[d + dim] - rect[d]) + y[d]
            )

        return Y_l_bounds + Y_u_bounds

    def F_J(self, learned_f, J, rect, lower_bounds, n=-3):
        """f: function, J: Jacobian matrix, rect: rectangle
        Given a rect return the smallest rectangle that contains the image of the
        J \cdot rect"""
        dim = int(len(rect) / 2)
        X = rect[0:dim]
        Y, S = learned_f(X)
        Y_max = Y + S*(2**n)
        Y_min = Y - S*(2**n)

        size_of_box = [rect[d+dim] - rect[d] for d in range(dim)]

        coordinate = [int(
            np.rint((rect[d] - lower_bounds[d]) / size_of_box[d])
        ) % 2 for d in range(dim)
        ]

        print(coordinate)
        Y_l_bounds = []
        Y_u_bounds = []

        print(f"X {X} \n Y {Y} \n Y_min {Y_min}\n Y_max {Y_max} \n size_of_box {size_of_box}")

        if any(coordinate):  # any even coordinate compute J

            Jac, _ = J(np.array(X).reshape(-1, dim))

            print(f"J {Jac}")
            for d in range(dim):
                J_d_norm = np.linalg.norm(Jac[d, :])
                Y_l_bounds.append(Y_min[:, d] - J_d_norm * size_of_box[d])
                Y_u_bounds.append(Y_max[:, d] + J_d_norm * size_of_box[d])

        else:

            for d in range(dim):
                Y_l_bounds.append(Y_min[:, d])
                Y_u_bounds.append(Y_max[:, d])

        return Y_l_bounds + Y_u_bounds

    def Box_ptwise(self, learned_f, rect, n=-3):
        """learned_f with predicted mean applied to the corner points
        and standard deviation applied to the center point"""

        dim = int(len(rect) / 2)
        X = CMGDB.CenterPoint(rect) + CMGDB.CornerPoints(rect)
        # Evaluate f at point in X

        Y, S = learned_f(X[0])

        Y = Y - S*(2**n)
        Y = np.concatenate((Y, Y + 2 * S * (2 ** n)))

        X = X[1::]
        for x_ in X:
            y_, _ = learned_f(x_)
            Y = np.concatenate((Y, y_))

        # Get lower and upper bounds of Y
        Y_l_bounds = [np.amin(Y[:, d]) for d in range(dim)]
        Y_u_bounds = [np.amax(Y[:, d]) for d in range(dim)]
        return Y_l_bounds + Y_u_bounds
