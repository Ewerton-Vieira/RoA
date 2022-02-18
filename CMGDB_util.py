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