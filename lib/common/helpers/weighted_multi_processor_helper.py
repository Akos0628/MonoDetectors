from lib.common.helpers.multi_processor_helper import normalize_angle
import numpy as np

def average_lists(list1, list2, x, slope):
    if len(list1) != len(list2):
        raise ValueError("A listák hossza nem egyezik.")
    if len(list1) == 0 or len(list2) == 0:
        return []

    averaged_list = []

    for row1, row2 in zip(list1, list2):
        if len(row1) != len(row2):
            raise ValueError("A sorok hossza nem egyezik.")

        averaged_row = []

        dist1 = np.linalg.norm([row1[8], row1[9], row1[10]])
        dist2 = np.linalg.norm([row2[8], row2[9], row2[10]])

        for idx, (val1, val2) in enumerate(zip(row1, row2)):
            if isinstance(val1, (float, np.float32)) and isinstance(val2, (float, np.float32)):
                averaged_row.append(weighted_average(val1, val2, dist1, dist2, x, slope))
            else:
                averaged_row.append(val1)

        rotation_y1 = row1[14]
        rotation_y2 = row2[14]
        avg_rotation_y = weighted_average_angle(rotation_y1, rotation_y2, dist1, dist2, x, slope)
        averaged_row[14] = avg_rotation_y

        averaged_list.append(averaged_row)

    return averaged_list
    
def close_weight(distance, x, slope): # better reversed thus MonoLSS is beter for close detection
    return 1 / (1 + np.exp(slope * (distance - x)))

def lss_weight(distance, x, slope):
    return close_weight(distance, x, slope)
def dtr_weight(distance, x, slope):
    return 1 - close_weight(distance, x, slope)

def weighted_average(value1, value2, dist1, dist2, x, slope):
    weight1 = lss_weight(dist1, x, slope)
    weight2 = dtr_weight(dist2, x, slope)
    total_weight = weight1 + weight2
    if total_weight > 0:
        return (weight1 * value1 + weight2 * value2) / total_weight
    return (value1 + value2) / 2

def weighted_average_angle(angle1, angle2, dist1, dist2, x, slope, threshold=np.pi / 2):
    angle1 = normalize_angle(angle1)
    angle2 = normalize_angle(angle2)
    delta = np.abs(normalize_angle(angle2 - angle1))

    if delta > threshold:
        angle2 = normalize_angle(angle2 + np.pi)

    weight1 = lss_weight(dist1, x, slope)
    weight2 = dtr_weight(dist2, x, slope)
    total_weight = weight1 + weight2

    if total_weight > 0:
        avg_cos = (weight1 * np.cos(angle1) + weight2 * np.cos(angle2)) / total_weight
        avg_sin = (weight1 * np.sin(angle1) + weight2 * np.sin(angle2)) / total_weight
    else:
        avg_cos = (np.cos(angle1) + np.cos(angle2)) / 2
        avg_sin = (np.sin(angle1) + np.sin(angle2)) / 2

    avg_angle = np.arctan2(avg_sin, avg_cos)
    return avg_angle