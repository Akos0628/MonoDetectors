import torch
import numpy as np

def convertAlpha2Rot(alpha, cx, P2):
    cx_p2 = P2[..., 0, 2]
    fx_p2 = P2[..., 0, 0]
    ry3d = alpha + np.arctan2(cx - cx_p2, fx_p2)
    ry3d[np.where(ry3d > np.pi)] -= 2 * np.pi
    ry3d[np.where(ry3d <= -np.pi)] += 2 * np.pi
    return ry3d


def convertRot2Alpha(ry3d, cx, P2):
    cx_p2 = P2[..., 0, 2]
    fx_p2 = P2[..., 0, 0]
    alpha = ry3d - np.arctan2(cx - cx_p2, fx_p2)
    alpha[alpha > np.pi] -= 2 * np.pi
    alpha[alpha <= -np.pi] += 2 * np.pi
    return alpha

def alpha2theta_3d(alpha, x, z, P2):
    """ Convert alpha to theta with 3D position
    Args:
        alpha [torch.Tensor/ float or np.ndarray]: size: [...]
        x     []: size: [...]
        z     []: size: [...]
        P2    [torch.Tensor/ np.ndarray]: size: [3, 4]
    Returns:
        theta []: size: [...]
    """
    offset = P2[0, 3] / P2[0, 0]
    if isinstance(alpha, torch.Tensor):
        theta = alpha + torch.atan2(x + offset, z)
    else:
        theta = alpha + np.arctan2(x + offset, z)
    return theta
