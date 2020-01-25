import numpy as np


def distance(src_x, src_y, dest_x, dest_y):
    return np.sqrt(np.square(src_x - dest_x) + np.square(src_y - dest_y))


def point_distance(matrix_pt_1, matrix_pt_2):
    return distance(matrix_pt_1[1], matrix_pt_1[0], matrix_pt_2[1], matrix_pt_2[0])
