from lib.monoDTR.visualDet3D.networks.lib.ops.iou3d.iou3d import boxes_iou3d_gpu
from lib.monoDTR.visualDet3D.networks.lib.fast_utils.bbox2d import iou_2d
from lib.common.helpers.detection_helper import detectionInfo
import numpy as np
import torch

def compareAll(list1, list2):
    comparison_matrix = []
    if len(list1) == 0 or len(list2) == 0:
        return []
    for item1 in list1:
        row = []
        for item2 in list2:
            result = compare(item1, item2)
            row.append(result)
        comparison_matrix.append(row)
    
    positions = []

    for row_idx, row in enumerate(comparison_matrix):
        max_value = max(row)
        if max_value != 0:
            max_pos = row.index(max_value)

            if all(max_value >= comparison_matrix[row_idx][j] for j in range(len(row))) and \
               all(max_value >= comparison_matrix[i][max_pos] for i in range(len(comparison_matrix))):
                positions.append((row_idx, max_pos))

    return positions

def compare(item1, item2):
    info1 = detectionInfo(item1)
    info2 = detectionInfo(item2)
    bbox3d1 = np.array([info1.tx, info1.ty, info1.tz, info1.h, info1.w, info1.l, info1.rot_global]).astype(np.float32)
    bbox3d2 = np.array([info2.tx, info2.ty, info2.tz, info2.h, info2.w, info2.l, info2.rot_global]).astype(np.float32)


    bbox1 = np.array([info1.xmin, info1.ymin, info1.xmax, info1.ymax]).astype(np.int32)
    bbox2 = np.array([info2.xmin, info2.ymin, info2.xmax, info2.ymax]).astype(np.int32)

    iou2d = iou_2d(np.array([bbox1]), np.array([bbox2]))
    #print(iou2d)
    if iou2d < 0.5:
        iou2d = 0.0
        iou3d = 0.0
    else:
        iou3d = boxes_iou3d_gpu(torch.tensor(np.array([bbox3d1])).cuda(), torch.tensor(np.array([bbox3d2])).cuda()).detach().cpu().numpy()[0][0]

    return iou3d + iou2d/2

def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def average_angle(angle1, angle2, threshold=np.pi / 2):
    angle1 = normalize_angle(angle1)
    angle2 = normalize_angle(angle2)

    delta = np.abs(normalize_angle(angle2 - angle1))

    # A két AI lehet, hogy ellentétes irányba tippelik meg az objektumokat, 
    # ezért ellenőrznünk kell, hogy a különbség nagyobb-e 90 foknál.
    # Ilyenkor a MonoLSS a baseline és a MonoDTR-t tükrözzük.
    if delta > threshold: 
        angle2 = normalize_angle(angle2 + np.pi)

    cos1, sin1 = np.cos(angle1), np.sin(angle1)
    cos2, sin2 = np.cos(angle2), np.sin(angle2)

    avg_cos = (cos1 + cos2) / 2
    avg_sin = (sin1 + sin2) / 2

    avg_angle = np.arctan2(avg_sin, avg_cos)
    return avg_angle

def average_lists(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("A listák hossza nem egyezik.")
    if len(list1) == 0 or len(list2) == 0:
        return []

    averaged_list = []

    for row1, row2 in zip(list1, list2):
        if len(row1) != len(row2):
            raise ValueError("A sorok hossza nem egyezik.")
        
        averaged_row = []
        for val1, val2 in zip(row1, row2):
            if isinstance(val1, (float, np.float32)) and isinstance(val2, (float, np.float32)):
                averaged_row.append((val1 + val2) / 2)
            
            else:
                averaged_row.append(val1)
        
        rotation_y1 = row1[14]
        rotation_y2 = row2[14]
        
        avg_rotation_y = average_angle(rotation_y1, rotation_y2)

        averaged_row[14] = avg_rotation_y
        
        averaged_list.append(averaged_row)

    return averaged_list