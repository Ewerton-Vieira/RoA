# RoA.py  # 2022-20-01
# MIT LICENSE 2020 Ewerton R. Vieira

import pychomp2 as pychomp
import CMGDB

import numpy as np
import time
from datetime import datetime
import math
import csv
import graphviz
import Poset_E
import os
import csv
import MultivaluedMap

# from pympler.asizeof import asizeof

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import matplotlib


class RoA:

    def propagate(self, u, adjacencies):
        """propagate a subtree with root u and
        assign the maximal morse node in the subtree for each tile"""

        adjacencies_morse_node = set()  # save morse node assigned to each adjacent tile
        for w in adjacencies:
            morse_node = self.dict_tiles.get(w, None)  # get the morse node assigned to tile

            if morse_node == None:  # if there isnt morse node assigned then propagate
                morse_node = self.propagate(w, self.map_graph.adjacencies(w))
            adjacencies_morse_node.add(morse_node)  # add the morse node assigned to tile w

        # get one of the maximal morse node in the subtree, it can have more than one,
        # so here we select the first ()
        # tiles that are mapped outside are assign to a fake node with value equal to -1

        max = list(self.MG.maximal(adjacencies_morse_node - {-1}))
        morse_node = max[0] if max else -1

        self.dict_tiles[u] = morse_node
        self.S.remove(u)  # remove it since we dont have to compute again
        return morse_node

    def assign_morse_nodes2tiles(self):
        """For each tile assign a morse node (creating a region of attraction for a downset)"""

        # clone self.tiles_in_morse_sets we dont have to recompute
        self.dict_tiles = dict(self.tiles_in_morse_sets)

        # remove keys in self.tiles_in_morse_sets we dont have to recompute
        self.S = list(self.vertices() - set(self.tiles_in_morse_sets.keys()))

        while self.S:  # loop: assign morse node to a tile and remove it
            v = self.S[0]
            # propagate to determine which morse node should be assign to tile v
            morse_node = self.propagate(v, self.map_graph.adjacencies(v))

        return self.dict_tiles

    def __init__(self, map_graph, morse_graph):
        """
        Region of Attraction class
        Assign cells in the phase space that are mapped to a unique Morse Node
        (Regions that are uniquely mapped to Morse Sets).
        Equivalent to an order retraction onto the Morse tiles by mapping
        to unique successor.
        """

        self.dir_path = os.path.abspath(os.getcwd()) + "/output/"

        self.morse_graph = morse_graph
        self.map_graph = map_graph

        # Get number of vertices
        self.num_verts = map_graph.num_vertices()

        self.vertices_ = {a for a in range(self.num_verts)}

        self.tiles_in_morse_sets = {}

        # create a dict: tile in morse set -> morse node
        # it is need to create the condensation graph
        for i in range(self.morse_graph.num_vertices()):
            for j in self.morse_graph.morse_set(i):

                self.tiles_in_morse_sets[j] = i

        MG = pychomp.DirectedAcyclicGraph()  # building Morse Graph Poset
        MG.add_vertex(0)
        for u in range(morse_graph.num_vertices()):
            for v in morse_graph.adjacencies(u):
                MG.add_edge(u, v)
        self.MG = Poset_E.Poset_E(MG)

        self.assign_morse_nodes2tiles()

        # print(
        #     f"memory size of: tiles_in_morse_sets={asizeof(self.tiles_in_morse_sets)}\n",
        #     f"memory size of: dict_tiles={asizeof(self.dict_tiles)}\n",
        #     f"MG={asizeof(self.MG)}")

    def vertices(self):
        """
        Return the set of elements in the poset
        """
        return self.vertices_

    def box_center(self, rect):
        dim = len(rect) // 2
        return [rect[i] + (rect[dim + i] - rect[i])/2 for i in range(dim)]

    def save_file(self, name=""):
        rect = self.morse_graph.phase_space_box(0)
        dim = int(len(rect) // 2)
        size_box = [rect[dim + i] - rect[i] for i in range(dim)]

        name = self.dir_path + name + "_RoA_" + ".csv"
        with open(name, "w") as file:
            f = csv.writer(file)
            f.writerow(["Box size"])
            f.writerow(size_box)
            f.writerow(["Tile", "Morse_node", "Box"])
            # tiles in roa
            for tile_in_roa in set(self.dict_tiles.items()) - set(self.tiles_in_morse_sets.items()):
                tile_in_roa = list(tile_in_roa) + \
                    [a for a in self.morse_graph.phase_space_box(tile_in_roa[0])]
                f.writerow(tile_in_roa)
            # tiles in morse sets
            f.writerow(["Tile_in_Morse_set", "Morse_node", "Box"])
            for tile_in_morse_set in self.tiles_in_morse_sets.items():
                tile_in_morse_set = list(tile_in_morse_set) + \
                    [a for a in self.morse_graph.phase_space_box(tile_in_morse_set[0])]
                f.writerow(tile_in_morse_set)

    def Morse_sets_vol(self):
        """Compute a dict that gives the volume of the regions of attraction"""

        d_vol = dict()
        tiles_and_morse_nodes = list(self.dict_tiles.items())
        for tile_and_morse in tiles_and_morse_nodes:
            i, j = tile_and_morse  # i is the tile and j is the associated morse node
            box = self.morse_graph.phase_space_box(i)

            size = len(box)
            half = int(size / 2)

            volume_cube = 1
            for k in range(half):
                volume_cube *= float(box[half + k]) - float(box[k])

            d_vol[j] = d_vol.get(j, 0) + volume_cube

        return d_vol

    def PlotTiles(self, selection=[], fig_w=8, fig_h=8, xlim=None, ylim=None,
                  cmap=matplotlib.cm.brg, name_plot=' ', from_file=None, plot_point=False, section=None):
        self.save_file(name="temp")

        rect = self.morse_graph.phase_space_box(0)
        dim = int(len(rect) // 2)

        # getting the bounds
        lower_bounds = rect[0:dim]
        upper_bounds = rect[dim::]

        for i in range(self.num_verts):
            box = self.morse_graph.phase_space_box(i)
            for index, j in enumerate(box[0:dim]):
                if lower_bounds[index] > j:
                    lower_bounds[index] = j
            for index, j in enumerate(box[dim::]):
                if upper_bounds[index] < j:
                    upper_bounds[index] = j

        fig, ax = PlotTiles(lower_bounds, upper_bounds, selection=selection, fig_w=fig_w, fig_h=fig_h, xlim=xlim,
                            ylim=ylim, cmap=cmap, name_plot=name_plot, from_file="temp", plot_point=plot_point, section=section)

        os.remove(self.dir_path + "temp_RoA_.csv")
        return fig, ax


def PlotTiles(lower_bounds, upper_bounds, selection=[], fig_w=8, fig_h=8, xlim=None, ylim=None,
              cmap=matplotlib.cm.brg, name_plot=' ', from_file=None, plot_point=False, section=None, from_file_basic=False):
    """ TODO:
    * section = ([z,w],(a,b,c,d)), 3D section when [z,w]=(c,d)
    * selection = selection of morse sets
    * check 1D and 3D plottings
    * check save file"""

    dim = len(lower_bounds)

    # path to save and read files
    dir_path = os.path.abspath(os.getcwd()) + "/output/"

    # read file saved by RoA
    if from_file and not from_file_basic:

        # read file
        from_file = dir_path + from_file + "_ROA_" + ".csv"
        with open(from_file, "r") as file:
            f = csv.reader(file, delimiter=',')
            next(f)
            box_size = [float(i) for i in next(f)]
            next(f)
            Tiles = []
            Morse_nodes = []
            Boxes = []
            num_morse_sets = 0
            counter_temp = 0
            for row in f:

                if row[0] == "Tile_in_Morse_set":
                    counter4morse_sets = counter_temp
                    continue
                counter_temp += 1
                Tiles.append(int(row[0]))
                Morse_nodes.append(int(row[1]))
                Boxes.append([float(a) for a in row[2:2+2*dim]])

                if num_morse_sets < int(row[1]):   # find the num_morse_sets - 1
                    num_morse_sets = int(row[1])

            num_morse_sets += 1

        # print(Tiles, Morse_nodes, Boxes)
        variables = [a for a in range(dim)]

        if not selection:
            selection = [i for i in range(num_morse_sets)]

        cmap_norm = matplotlib.colors.Normalize(vmin=0, vmax=num_morse_sets-1)

        morse = {}  # tiles in morse sets
        tiles = {}  # tiles in regions of attraction (not including tiles in morse set)

        volume_cube = 1
        d_vol = dict()
        for i in range(dim):
            volume_cube *= box_size[i]

        for i, m_node in enumerate(Morse_nodes):
            if m_node not in selection:  # only add the selected Morse sets
                continue
            clr = matplotlib.colors.to_hex(cmap(cmap_norm(m_node)))

            if i < counter4morse_sets:  # associate center of boxes to the Morse tiles
                B = tiles.get(clr, [])
                B.append(Boxes[i])
                tiles[clr] = B
                # compute the total volume of Morse tiles
                d_vol[m_node] = d_vol.get(m_node, 0) + volume_cube

            else:  # associate  boxes to the Morse sets
                A = morse.get(clr, [])
                A.append(Boxes[i])
                morse[clr] = A

            # # compute the total volume of Morse tiles
            # d_vol[m_node] = d_vol.get(m_node, 0) + volume_cube

        print(f'dictionary with volume of all Morse tiles = {d_vol}')
    # read file saved by RoA

    # read file saved by CMGDB (only Morse tiles)
    if from_file and from_file_basic:

        from_file = dir_path + from_file + ".csv"
        morse = {}
        with open(from_file, "r") as file:
            f = csv.reader(file, delimiter=',')
            Morse_nodes = []
            box = []
            for row in f:
                dim = len(row)//2
                Morse_nodes.append(int(float(row[-1])))
                box.append([float(row[i]) for i in range(2*dim)])
            box_size = [float(row[i+dim]) - float(row[i]) + 0.000000005 for i in range(dim)]

        variables = [a for a in range(dim)]
        num_morse_sets = Morse_nodes[-1] + 1

        cmap_norm = matplotlib.colors.Normalize(vmin=0, vmax=num_morse_sets-1)

        morse = {}
        tiles = {}
        for i, m_node in enumerate(Morse_nodes):
            clr = matplotlib.colors.to_hex(cmap(cmap_norm(m_node)))
            A = morse.get(clr, [])
            A.append(box[i])
            morse[clr] = A

        tiles = morse
    # read file saved by CMGDB (only Morse tiles)

    # for dim 1, add fake dimension
    if dim == 1:
        d2 = 0
        box_size.append(fig_h/32)

    # 2D plotting or 2D with a given section
    if section or dim == 2:
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)

        if dim == 2:
            section = ([], 'projection')

        variables_section = list(set(variables) - set(section[0]))
        d1 = variables_section[0]
        d2 = variables_section[1]

        if section[1] == 'projection':
            section = ([], 'projection')  # clean section to do projection

        if not from_file_basic:
            for i, j in tiles.items():
                rectangles_list = []  # set instead of list for avoiding repetition
                for row in j:
                    if section[1] == 'projection':
                        in_section = [True]

                    else:
                        in_section = [row[k] - box_size[k]/2 <= section[1][k]
                                      < row[k] + box_size[k]/2 for k in section[0]]

                    if all(in_section):
                        rectangle = Rectangle((row[d1], row[d2]), box_size[d1], box_size[d2])
                        rectangles_list.append(rectangle)
                pc = PatchCollection(rectangles_list, cmap=cmap, fc=i, alpha=0.4, ec='none')
                ax.add_collection(pc)

        for i, j in morse.items():
            rectangles_list = []
            for row in j:

                if section[1] == 'projection':
                    in_section = [True]

                else:
                    in_section = [row[k] - box_size[k]/2 <= section[1][k]
                                  < row[k] + box_size[k]/2 for k in section[0]]

                if all(in_section):
                    rectangle = Rectangle((row[d1], row[d2]), box_size[d1], box_size[d2])
                    rectangles_list.append(rectangle)
            pc = PatchCollection(rectangles_list, cmap=cmap, fc=i, alpha=1, ec='none')
            ax.add_collection(pc)
        if xlim and ylim:
            ax.set_xlim([xlim[0], xlim[1]])
            ax.set_ylim([ylim[0], ylim[1]])
        else:
            ax.set_xlim([lower_bounds[d1], upper_bounds[d1]])
            ax.set_ylim([lower_bounds[d2], upper_bounds[d2]])
        ax.set_xlabel(str(d1))
        ax.set_ylabel(str(d2))
        if section[1] == 'projection':
            value_section = tuple([0 for i in section[0]])

        else:
            value_section = tuple([int(section[1][i]*100) for i in section[0]])
        name_plot = f'{dir_path}{name_plot}_{section[0]}_{value_section}'
    # 2D plotting or 2D with a given section

    # 3D plotting
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        #

        if plot_point:
            for i, j in morse.items():
                j = np.array(j)
                plt.plot(j[:, 0], j[:, 1], j[:, 2], '.', color=i)

            # for i, j in tiles.items():
            #     j = np.array(j)
            #     plt.plot(j[:, 0], j[:, 1], j[:, 2], '.', color=i, alpha=0.2)

        else:

            voxel_grid_x = int(np.rint((upper_bounds[0]-lower_bounds[0])/box_size[0]))
            voxel_grid_y = int(np.rint((upper_bounds[1]-lower_bounds[1])/box_size[1]))
            voxel_grid_z = int(np.rint((upper_bounds[2]-lower_bounds[2])/box_size[2]))

            voxel_grid = np.zeros((voxel_grid_x, voxel_grid_y, voxel_grid_z))

            x, y, z = np.indices(np.array(voxel_grid.shape)+1)

            x = box_size[0]*x + lower_bounds[0]
            y = box_size[1]*y + lower_bounds[1]
            z = box_size[2]*z + lower_bounds[2]

            # for i, j in tiles.items():
            #     for row in j:
            #         v_x = int(np.rint((row[0] - box_size[0]/2 - lower_bounds[0]) / box_size[0]))
            #         v_y = int(np.rint((row[1] - box_size[1]/2 - lower_bounds[1]) / box_size[1]))
            #         v_z = int(np.rint((row[2] - box_size[2]/2 - lower_bounds[2]) / box_size[2]))
            #         voxel_grid[v_x, v_y, v_z] = True
            #     # ax.voxels(x, y, z, voxel_grid, facecolors=i, alpha=0.2)
            #     voxel_grid = np.where(voxel_grid == True, False, True)
            #     ax.voxels(x, y, z, voxel_grid, facecolors=i, alpha=0.2)
            #     voxel_grid = np.zeros((voxel_grid_x, voxel_grid_y, voxel_grid_z))

            for i, j in morse.items():
                for row in j:
                    v_x = int(np.rint((row[0] - box_size[0]/2 - lower_bounds[0]) / box_size[0]))
                    v_y = int(np.rint((row[1] - box_size[1]/2 - lower_bounds[1]) / box_size[1]))
                    v_z = int(np.rint((row[2] - box_size[2]/2 - lower_bounds[2]) / box_size[2]))
                    voxel_grid[v_x, v_y, v_z] = True
                # ax.voxels(x, y, z, voxel_grid, facecolors=i, alpha=0.2)
                voxel_grid = np.where(voxel_grid == True, True, True)
                ax.voxels(x, y, z, voxel_grid, facecolors=i, alpha=0.8)
                voxel_grid = np.zeros((voxel_grid_x, voxel_grid_y, voxel_grid_z))

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Theta')
        name_plot = dir_path + name_plot

    # 3D plotting

    # save file with name_plot
    if name_plot != ' ':
        plt.savefig(name_plot)

    return fig, ax
