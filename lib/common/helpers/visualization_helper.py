import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib.gridspec import GridSpec

from lib.common.helpers.detection_helper import detectionInfo

def visualization(image, calib, detections, drawBird):
    P2 = calib[0].P2

    fig = plt.figure(figsize=(20.00, 5.12), dpi=100)

    # fig.tight_layout()
    gs = GridSpec(1, 4)
    gs.update(wspace=0)  # set the spacing between axes.

    ax = fig.add_subplot(gs[0, :3])
    if drawBird:
        ax2 = fig.add_subplot(gs[0, 3:])

    shape = 900

    for line_p in detections:
        obj = detectionInfo(line_p)
        truncated = np.abs(float(obj.truncation))
        occluded = np.abs(float(obj.occlusion))
        trunc_level = 255

        # truncated object in dataset is not observable
        if truncated < trunc_level:
            color = 'green'
            if obj.name == 'Cyclist':
                color = 'yellow'
            elif obj.name == 'Pedestrian':
                color = 'cyan'
            
            #draw_2Dbox(ax, obj, color)
            draw_3Dbox(ax, P2, obj, color)
            if drawBird:
                draw_birdeyes(ax2, obj, shape, color)

    # visualize 3D bounding box
    ax.imshow(image)
    ax.set_xticks([]) #remove axis value
    ax.set_yticks([])

    if drawBird:
        birdimage = np.zeros((shape, shape, 3), np.uint8)
        # plot camera view range
        x1 = np.linspace(0, shape / 2)
        x2 = np.linspace(shape / 2, shape)
        ax2.plot(x1, shape / 2 - x1, ls='--', color='grey', linewidth=1, alpha=0.5)
        ax2.plot(x2, x2 - shape / 2, ls='--', color='grey', linewidth=1, alpha=0.5)
        ax2.plot(shape / 2, 0, marker='+', markersize=16, markeredgecolor='red')

        # visualize bird eye view
        ax2.imshow(birdimage, origin='lower')
        ax2.set_xticks([])
        ax2.set_yticks([])

    plt.show()

def compute_birdviewbox(obj, shape, scale):
    h = obj.h * scale
    w = obj.w * scale
    l = obj.l * scale
    x = obj.tx * scale
    y = obj.ty * scale
    z = obj.tz * scale
    rot_y = obj.rot_global

    R = np.array([[-np.cos(rot_y), np.sin(rot_y)],
                  [np.sin(rot_y), np.cos(rot_y)]])
    t = np.array([x, z]).reshape(1, 2).T

    x_corners = [0, l, l, 0]  # -l/2
    z_corners = [w, w, 0, 0]  # -w/2

    x_corners += -w / 2
    z_corners += -l / 2

    # bounding box in object coordinate
    corners_2D = np.array([x_corners, z_corners])
    # rotate
    corners_2D = R.dot(corners_2D)
    # translation
    corners_2D = t - corners_2D
    # in camera coordinate
    corners_2D[0] += int(shape/2)
    corners_2D = (corners_2D).astype(np.int16)
    corners_2D = corners_2D.T

    return np.vstack((corners_2D, corners_2D[0,:]))

def draw_birdeyes(ax2, obj, shape, color):
    scale = 15
    pred_corners_2d = compute_birdviewbox(obj, shape, scale)
    codes = [Path.LINETO] * pred_corners_2d.shape[0]
    codes[0] = Path.MOVETO
    codes[-1] = Path.CLOSEPOLY
    pth = Path(pred_corners_2d, codes)
    p = patches.PathPatch(pth, fill=False, color=color)
    ax2.add_patch(p)

def draw_2Dbox(ax, obj, color):
    xmin = obj.xmin
    xmax = obj.xmax
    ymin = obj.ymin
    ymax = obj.ymax
    width = xmax - xmin
    height = ymax - ymin

    ax.add_patch(patches.Rectangle((xmin, ymin), width, height, fill=False, color='red'))

def compute_3Dbox(P2, obj):
    R = np.array([[np.cos(obj.rot_global), 0, np.sin(obj.rot_global)],
                  [0, 1, 0],
                  [-np.sin(obj.rot_global), 0, np.cos(obj.rot_global)]])

    x_corners = [0, obj.l, obj.l, obj.l, obj.l, 0, 0, 0]  # -l/2
    y_corners = [0, 0, obj.h, obj.h, 0, 0, obj.h, obj.h]  # -h
    z_corners = [0, 0, 0, obj.w, obj.w, obj.w, obj.w, 0]  # -w/2

    x_corners = [i - obj.l / 2 for i in x_corners]
    y_corners = [i - obj.h for i in y_corners]
    z_corners = [i - obj.w / 2 for i in z_corners]

    corners_3D = np.array([x_corners, y_corners, z_corners])
    corners_3D = R.dot(corners_3D)
    corners_3D += np.array([obj.tx, obj.ty, obj.tz]).reshape((3, 1))

    corners_3D_1 = np.vstack((corners_3D, np.ones((corners_3D.shape[-1]))))
    corners_2D = P2.dot(corners_3D_1)
    corners_2D = corners_2D / corners_2D[2]
    corners_2D = corners_2D[:2]

    return corners_2D

def draw_3Dbox(ax, P2, obj, color):
    corners_2D = compute_3Dbox(P2, obj)

    # draw all lines through path
    # https://matplotlib.org/users/path_tutorial.html
    bb3d_lines_verts_idx = [0, 1, 2, 3, 4, 5, 6, 7, 0, 5, 4, 1, 2, 7, 6, 3]
    bb3d_on_2d_lines_verts = corners_2D[:, bb3d_lines_verts_idx]
    verts = bb3d_on_2d_lines_verts.T
    codes = [Path.LINETO] * verts.shape[0]
    codes[0] = Path.MOVETO
    # codes[-1] = Path.CLOSEPOLYq
    pth = Path(verts, codes)
    p = patches.PathPatch(pth, fill=False, color=color, linewidth=2)

    width = corners_2D[:, 3][0] - corners_2D[:, 1][0]
    height = corners_2D[:, 2][1] - corners_2D[:, 1][1]
    # put a mask on the front
    front_fill = patches.Rectangle((corners_2D[:, 1]), width, height, fill=True, color=color, alpha=0.4)
    ax.add_patch(p)
    ax.add_patch(front_fill)