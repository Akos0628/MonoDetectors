import cv2
import torch
import numpy as np
from copy import deepcopy

from lib.monoDTR.visualDet3D.networks.utils.utils import BackProjection
from lib.monoDTR.visualDet3D.networks.utils.utils import BBox3dProjector

class Printer(object):
    def __init__(self, cfg, model, logger):
        self.cfg = cfg
        self.model = model
        self.model.cuda()

        self.logger = logger
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if self.cfg.get('resume_model', None):
            print(cfg['resume_model'])
            state_dict = torch.load(cfg['resume_model'], map_location='cuda:0', weights_only=False)
            new_dict = state_dict.copy()
            self.model.load_state_dict(new_dict, strict=False)
        
        self.model.eval()
        self.model.to(self.device)
        self.backprojector = BackProjection().cuda()
        self.projector = BBox3dProjector().cuda()

    def print(self, idx, img, calibs, resolution, downsample, mean, std, cls_mean_size):
        torch.set_grad_enabled(False)

        original_P2 = calibs[0].P2
        P2 = deepcopy(original_P2)
        rgb_images, P2 = self.toMonoDTR(img, P2)

        scores, bbox, obj_index = self.model(
            [
                torch.from_numpy(rgb_images).float().cuda().float().contiguous(), 
                torch.tensor(np.array([P2])).float().cuda().float()
            ]
        )
        obj_types = [self.cfg['obj_types'][i.item()] for i in obj_index]

        bbox_2d = bbox[:, 0:4]
        if bbox.shape[1] > 4: # run 3D
            bbox_3d_state = bbox[:, 4:] #[cx,cy,z,w,h,l,alpha, bot, top]
            bbox_3d_state_3d = self.backprojector(bbox_3d_state, P2) #[x, y, z, w,h ,l, alpha, bot, top]

            _, _, thetas = self.projector(bbox_3d_state_3d, bbox_3d_state_3d.new(P2))

            scale_x = original_P2[0, 0] / P2[0, 0]
            scale_y = original_P2[1, 1] / P2[1, 1]
            
            shift_left = original_P2[0, 2] / scale_x - P2[0, 2]
            shift_top  = original_P2[1, 2] / scale_y - P2[1, 2]
            bbox_2d[:, 0:4:2] += shift_left
            bbox_2d[:, 1:4:2] += shift_top

            bbox_2d[:, 0:4:2] *= scale_x
            bbox_2d[:, 1:4:2] *= scale_y

            return toPredictions(scores, bbox_2d, bbox_3d_state_3d, thetas, obj_types)
        else:
            print('non 3D')
    

    def toMonoDTR(self, img, calib):
        rgb_images = img

        rgb_images = np.array(rgb_images).astype(np.float32)
        rgb_images, calib = cropTop(rgb_images, calib, 100)
        rgb_images, calib = resize(rgb_images, calib, (288, 1280))
        rgb_images, calib = normalize(rgb_images, calib)

        rgb_images = np.array([rgb_images])#[batch, H, W, 3]
        rgb_images = rgb_images.transpose([0, 3, 1, 2])

        return rgb_images, calib
    
def toPredictions(scores, bbox_2d, bbox_3d_state_3d, thetas, obj_types, threshold=0.9): 
    scores = scores.detach().cpu().numpy()
    bbox_2d = bbox_2d.detach().cpu().numpy()
    bbox_3d_state_3d = bbox_3d_state_3d.detach().cpu().numpy()
    thetas = thetas.detach().cpu().numpy()
    
    preds = []
    if bbox_3d_state_3d is None:
        bbox_3d_state_3d = np.ones([bbox_2d.shape[0], 7], dtype=int)
        bbox_3d_state_3d[:, 3:6] = -1
        bbox_3d_state_3d[:, 0:3] = -1000
        bbox_3d_state_3d[:, 6]   = -10
    else:
        for i in range(len(bbox_2d)):
            bbox_3d_state_3d[i][1] = bbox_3d_state_3d[i][1] + 0.5*bbox_3d_state_3d[i][4] # kitti receive bottom center

    if thetas is None:
        thetas = np.ones(bbox_2d.shape[0]) * -10
    if len(scores) > 0:
        for i in range(len(bbox_2d)):
            if scores[i] < threshold:
                continue
            bbox = bbox_2d[i]
            preds.append([obj_types[i], 0, 0, bbox_3d_state_3d[i][-1], bbox[0], bbox[1], bbox[2], bbox[3],
                bbox_3d_state_3d[i][4], bbox_3d_state_3d[i][3], bbox_3d_state_3d[i][5],
                bbox_3d_state_3d[i][0], bbox_3d_state_3d[i][1], bbox_3d_state_3d[i][2],
                thetas[i], scores[i]])
    return preds


def normalize(left_image, p2, mean=np.array([0.485, 0.456, 0.406]), stds=np.array([0.229, 0.224, 0.225])):
    left_image = left_image.astype(np.float32)
    left_image /= 255.0
    left_image -= np.tile(mean, int(left_image.shape[2]/mean.shape[0]))
    left_image /= np.tile(stds, int(left_image.shape[2]/stds.shape[0]))
    left_image.astype(np.float32)
    return left_image, p2

def resize(left_image, p2, size=None, preserve_aspect_ratio=True):
    if preserve_aspect_ratio:
        scale_factor = size[0] / left_image.shape[0]

        h = np.round(left_image.shape[0] * scale_factor).astype(int)
        w = np.round(left_image.shape[1] * scale_factor).astype(int)
        
        scale_factor_yx = (scale_factor, scale_factor)
    else:
        scale_factor_yx = (size[0] / left_image.shape[0], size[1] / left_image.shape[1])

        h = size[0]
        w = size[1]

    # resize
    left_image = cv2.resize(left_image, (w, h))

    if len(size) > 1:

        # crop in
        if left_image.shape[1] > size[1]:
            left_image = left_image[:, 0:size[1], :]
        # pad out
        elif left_image.shape[1] < size[1]:
            padW = size[1] - left_image.shape[1]
            left_image  = np.pad(left_image,  [(0, 0), (0, padW), (0, 0)], 'constant')

    if p2 is not None:
        p2[0, :]   = p2[0, :] * scale_factor_yx[1]
        p2[1, :]   = p2[1, :] * scale_factor_yx[0]
    
    return left_image, p2

def cropTop(left_image, p2, crop_top_index=None, output_height=None):
        height, width = left_image.shape[0:2]

        if crop_top_index is not None:
            h_out = height - crop_top_index
            upper = crop_top_index
        else:
            h_out = output_height
            upper = height - output_height
        lower = height

        left_image = left_image[upper:lower]
        ## modify calibration matrix
        if p2 is not None:
            p2[1, 2] = p2[1, 2] - upper               # cy' = cy - dv
            p2[1, 3] = p2[1, 3] - upper * p2[2, 3] # ty' = ty - dv * tz

        return left_image, p2
