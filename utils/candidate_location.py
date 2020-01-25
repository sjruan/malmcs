from utils.distance import distance
import numpy as np


def get_candidate_locations_general(cur_loc, target_loc, distance, row_num, col_num):
    if cur_loc[0] == target_loc[0] and cur_loc[1] == target_loc[1]:
        return get_candidate_locations(cur_loc, distance / 2.0, row_num, col_num)
    else:
        return get_candidate_locations_ellipse(cur_loc, target_loc, distance, row_num, col_num)


def get_candidate_locations(cur_location, radius, row_num, col_num):
    """
    get candidate locations within distance
    :param cur_location:
    :param distance:
    :param row_num:
    :param col_num
    :return:
    """
    cur_y, cur_x = cur_location
    delta = int(radius)
    max_x = cur_x + delta if cur_x + delta < col_num else col_num - 1
    min_x = cur_x - delta if cur_x - delta >= 0 else 0
    max_y = cur_y + delta if cur_y + delta < row_num else row_num - 1
    min_y = cur_y - delta if cur_y - delta >= 0 else 0
    candidates = []
    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            if distance(cur_x, cur_y, x, y) < radius:
                candidates.append((y, x))
    return candidates


def get_candidate_locations_ellipse(f1, f2, major_axis, row_num, col_num):
    """
    get candidate locations within ellipse
    :param f1: focus point
    :param f2: focus point
    :param major_axis:
    :param row_num:
    :param col_num:
    :return:
    """
    f1_y, f1_x = f1
    f2_y, f2_x = f2
    a = major_axis / 2.0
    c = distance(f1_x, f1_y, f2_x, f2_y) / 2.0
    b = np.sqrt(a * a - c * c)
    major_bound_1 = ((f2_x - f1_x) * (a + c) / (2 * c) + f1_x, (f2_y - f1_y) * (a + c) / (2 * c) + f1_y)
    major_bound_2 = ((f1_x - f2_x) * (a + c) / (2 * c) + f2_x, (f1_y - f2_y) * (a + c) / (2 * c) + f2_y)
    delta_x = b * (abs(major_bound_1[1] - major_bound_2[1])) / (2 * a)
    if f1_x == f2_x:
        delta_y = 0
    else:
        delta_y = np.sqrt(b * b - delta_x * delta_x)
    p1 = (major_bound_1[0] + delta_x, major_bound_1[1] - delta_y)
    p2 = (major_bound_1[0] - delta_x, major_bound_1[1] + delta_y)
    p3 = (major_bound_2[0] - delta_x, major_bound_2[1] + delta_y)
    p4 = (major_bound_2[0] + delta_x, major_bound_2[1] - delta_y)
    min_x = int(max(min([p1[0], p2[0], p3[0], p4[0]]), 0))
    max_x = int(min(max([p1[0], p2[0], p3[0], p4[0]]), col_num - 1))
    min_y = int(max(min([p1[1], p2[1], p3[1], p4[1]]), 0))
    max_y = int(min(max([p1[1], p2[1], p3[1], p4[1]]), row_num - 1))
    candidates = []
    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            if distance(f1_x, f1_y, x, y) + distance(x, y, f2_x, f2_y) < major_axis:
                candidates.append((y, x))
    return candidates
