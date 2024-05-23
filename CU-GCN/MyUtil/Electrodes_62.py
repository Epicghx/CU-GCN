import numpy as np
import math as m
from matplotlib import pyplot as plt
from einops import rearrange

'''
62 channels 
'''


def main():
    elec = Electrodes()
    adj = elec.adjacency_matrix
    # print(elec.adjacency_matrix)


class Electrodes:
    def __init__(self, add_global_connections=True, expand_3d=False):
        # X, Y, Z coordinates of the electrodes
        self.positions_3d = np.array([[-27, 83, -3], [-36, 76, 24],
                                      [-25, 62, 56], [-48, 59, 44], [-64, 55, 23], [-71, 51, -3],
                                      [-83, 27, -3], [-78, 30, 27], [-59, 31, 56], [-33, 33, 74],
                                      [-34, 0, 81], [-63, 0, 61], [-82, 0, 31], [-87, 0, -3],
                                      [-83, -27, -3], [-78, -30, 27], [-59, -31, 56], [-33, -33, 74],
                                      [-25, -62, 56], [-48, -59, 44], [-64, -55, 23], [-71, -51, -3],
                                      [-51, -71, -3], [-44, -72, 13], [-36, -76, 24], [-27, -83, -3], [-36, -87, -37],
                                      [0, -87, -3], [0, -82, 31], [0, -63, 61], [0, -34, 81],

                                      [0, 87, -3], [27, 83, -3], [36, 76, 24],
                                      [0, 63, 61], [25, 62, 56], [48, 59, 44], [64, 55, 23], [71, 51, -3],
                                      [83, 27, -3], [78, 30, 27], [59, 31, 56], [33, 33, 74], [0, 34, 81],
                                      [0, 0, 88], [34, 0, 81], [63, 0, 61], [82, 0, 31], [87, 0, -3],
                                      [83, -27, -3], [78, -30, 27], [59, -31, 56], [33, -33, 74],
                                      [25, -62, 56], [48, -59, 44], [64, -55, 23], [71, -51, -3],
                                      [51, -71, -3], [44, -72, 13], [36, -76, 24], [27, -83, -3], [36, -87, -37]])
        self.channel_names = np.array(['Fp1', 'AF3',
                                       'F1', 'F3', 'F5', 'F7',
                                       'FT7', 'FC5', 'FC3', 'FC1',
                                       'C1', 'C3', 'C5', 'T7',
                                       'TP7', 'CP5', 'CP3', 'CP1',
                                       'P1', 'P3', 'P5', 'P7',
                                       'PO7', 'PO5', 'PO3', 'O1', 'CB1',  # PO5, CB1
                                       'Oz', 'POz', 'Pz', 'CPz',

                                       'FPz', 'Fp2', 'AF4',
                                       'Fz', 'F2', 'F4', 'F6', 'F8',
                                       'FT8', 'FC6', 'FC4', 'FC2', 'FCz'
                                                                   'Cz', 'C2', 'C4', 'C6', 'T8',
                                       'TP8', 'CP6', 'CP4', 'CP2',
                                       'P2', 'P4', 'P6', 'P8',
                                       'PO8', 'PO6', 'PO4', 'O2', 'CB2'  # PO6, CB2
                                       ])
        # print(self.channel_names)
        # Global connections will get a weight of -1 in the adj matrix
        self.global_connections = np.array(
            [['Fp1', 'Fp2'], ['AF3', 'AF4'], ['F5', 'F6'], ['FC5', 'FC6'], ['C5', 'C6'], ['CP5', 'CP6'], ['P5', 'P6'],
             ['PO5', 'PO6'], ['O1', 'O2']])
        self.positions_2d = self.get_proyected_2d_positions()
        self.adjacency_matrix = self.get_adjacency_matrix(add_global_connections)
        if expand_3d:
            print('Augmenting datapoints by interpolation')
            self.original_positions_3d = self.positions_3d.copy()
            self.positions_3d = self.generate_in_between_positions(num_points=1, verbose=False)
            self.adjacency_matrix = self.get_adjacency_matrix(add_global_connections=True,
                                                              positions_3d=self.positions_3d)

    # Helper function for get_proyected_2d_positions
    def azim_proj(self, pos):
        [r, elev, az] = self.cart2sph(pos[0], pos[1], pos[2])
        return self.pol2cart(az, m.pi / 2 - elev)

    # Helper function for get_proyected_2d_positions
    def cart2sph(self, x, y, z):
        x2_y2 = x ** 2 + y ** 2
        r = m.sqrt(x2_y2 + z ** 2)  # r     tant^(-1)(y/x)
        elev = m.atan2(z, m.sqrt(x2_y2))  # Elevation
        az = m.atan2(y, x)  # Azimuth
        return r, elev, az

    # Helper function for get_proyected_2d_positions
    def pol2cart(self, theta, rho):
        return rho * m.cos(theta), rho * m.sin(theta)

    def get_proyected_2d_positions(self, positions_3d=None):
        positions_3d = self.positions_3d if positions_3d is None else positions_3d
        pos_2d = np.array([self.azim_proj(pos_3d) for pos_3d in positions_3d])
        return pos_2d

    # Distance using projected coordinates
    # Testing needed. What is this distance when seen in 3-D? Arc of the circunference?
    def get_projected_2d_distance(self, name1, name2):
        index1, index2 = np.where(self.channel_names == name1)[0][0], np.where(self.channel_names == name2)[0][0]
        p1, p2 = self.positions_2d[index1], self.positions_2d[index2]
        incX, incY = p1[0] - p2[0], p1[1] - p2[1]
        return m.sqrt(incX ** 2 + incY ** 2)

    def plot_2d_projection(self, points=np.array([])):
        points = self.positions_2d if points.size == 0 else points
        fig, ax = plt.subplots()
        ax.scatter(points[:, 0], points[:, 1])
        for i, name in enumerate(self.channel_names):
            plt.text(points[:, 0][i], points[:, 1][i], name)
        plt.show()

    # Distance using 3d positions
    def get_3d_distance(self, name1, name2):
        index1, index2 = np.where(self.channel_names == name1)[0][0], np.where(self.channel_names == name2)[0][0]
        p1, p2 = self.positions_3d[index1], self.positions_3d[index2]
        incX, incY, incZ = p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]
        return m.sqrt(incX ** 2 + incY ** 2 + incZ ** 2)

    # get_adjacency_matrix is the main method for the Electrodes class. It returns a fixed adjacency matrix for the graph based on the 3-d coordinates of the electrodes
    # Symetrical, contains self-loops (learnable [?])
    # Calibration constant should keep 20% of the links acording to the paper
    def get_adjacency_matrix(self, add_global_connections, positions_3d=np.array([]), calibration_constant=5,
                             active_threshold=0.1):
        positions_3d = self.positions_3d if positions_3d.size == 0 else positions_3d
        # Expand 3d position vector to a 32*32 matrix
        distance_3d_matrix = np.array([positions_3d, ] * len(positions_3d))
        # Transpose
        distance_3d_matrix = rearrange(distance_3d_matrix, 'h w d -> w h d')
        # Calculate 3d distances (m.sqrt(incX**2 + incY**2 + incZ**2))
        distance_3d_matrix = distance_3d_matrix - positions_3d
        distance_3d_matrix = distance_3d_matrix ** 2
        distance_3d_matrix = distance_3d_matrix.sum(axis=-1)
        distance_3d_matrix = np.sqrt(distance_3d_matrix)
        # Define local connections
        distance_3d_matrix[distance_3d_matrix != 0] = (
                    calibration_constant / distance_3d_matrix[distance_3d_matrix != 0])

        if add_global_connections:
            global_indices = [[np.where(self.channel_names == e[0])[0], np.where(self.channel_names == e[1])[0]] for e
                              in self.global_connections]
            global_indices = np.array(global_indices).squeeze()
            # Set symetric global connections
            distance_3d_matrix[global_indices[:, 0], global_indices[:, 1]] -= 1
            distance_3d_matrix[global_indices[:, 1], global_indices[:, 0]] -= 1

        local_conn_mask = distance_3d_matrix > active_threshold
        local_connections = distance_3d_matrix * local_conn_mask
        # Min max normalize connections and initialice adjacency_matrix
        np.fill_diagonal(local_connections, 0)
        adj_matrix = local_connections
        adj_matrix = (local_connections - local_connections.min()) / (local_connections.max() - local_connections.min())
        # Add self-loops
        np.fill_diagonal(adj_matrix, 1)
        # Global connections get initialised to -1


        return adj_matrix  # adj_matrix = local connections + self-loops + (optional) global connections

    def generate_in_between_positions(self, num_points=1, verbose=False):
        """
        Returns new positions in-between current electrodes with a link between them
        :param positions_3d: List of 3D electrode positional information [x,y,z]
        :param adjacency_matrix: Graph adjacency matrix. Will determine where to include the new points
        :param num_points: Number of points to include for each value in the adjacency matrix > 0
        :return: List of positions [x,y,z]
        """
        if num_points < 1:
            return self.positions_3d
        # Assumes symmetric adj.matrix
        new_points = []
        for i in range(32):
            for j in range(i, 32):
                a = self.adjacency_matrix[i, j]
                if i != j and a != 0 and a != -1:
                    if verbose:
                        print(self.channel_names[i], self.channel_names[j], a)
                        print(self.positions_3d[i], self.positions_3d[j])
                    inc = (self.positions_3d[i] - self.positions_3d[j])
                    inc_step = inc / (num_points + 1)
                    for n in range(1, num_points + 1):
                        new_point = self.positions_3d[j] + (n * inc_step)
                        if verbose:
                            print(new_point)
                        new_points.append(new_point)
        return np.vstack([self.positions_3d, np.array(new_points)])


if __name__ == '__main__':
    main()