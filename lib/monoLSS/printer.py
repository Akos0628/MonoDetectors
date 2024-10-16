import torch
import numpy as np
from PIL import Image

from lib.monoLSS.helpers.save_helper import load_checkpoint
from lib.monoLSS.helpers.decode_helper import extract_dets_from_outputs
from lib.monoLSS.helpers.decode_helper import decode_detections
from lib.common.datasets.kitti_utils import get_affine_transform

class Printer(object):
    def __init__(self, cfg, model, logger):
        self.cfg = cfg
        self.model = model
        self.model.eval()

        self.logger = logger
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if self.cfg.get('resume_model', None):
            load_checkpoint(
                model = self.model,
                optimizer = None,
                filename = cfg['resume_model'],
                logger = self.logger,
                map_location=self.device
            )
        self.model.to(self.device)

    def print(self, idx, img, calibs, resolution, downsample, mean, std, cls_mean_size):
        img_size = np.array(img.size)

        center = np.array(img_size) / 2
        crop_size = img_size
        coord_range = np.array([center-crop_size/2,center+crop_size/2]).astype(np.float32)

        trans, trans_inv = get_affine_transform(center, crop_size, 0, resolution, inv=1)
        inputs = img.transform(tuple(resolution.tolist()),
                            method=Image.AFFINE,
                            data=tuple(trans_inv.reshape(-1).tolist()),
                            resample=Image.BILINEAR)
        inputs = np.array(inputs).astype(np.float32) / 255.0
        #inputs = (inputs - mean) / std
        inputs = inputs.transpose(2, 0, 1)  # C * H * W

        features_size = resolution // downsample
        info = {'img_id': np.asarray([idx]),
                'img_size': np.asarray([img_size]),
                'bbox_downsample_ratio': np.asarray([img_size/features_size])}
        
        torch.set_grad_enabled(False)
        inputs = torch.Tensor(inputs).unsqueeze(0).to(self.device)
        calib = torch.Tensor(calibs[0].P2).unsqueeze(0).to(self.device)
        coord_ranges = torch.Tensor(coord_range).unsqueeze(0).to(self.device)
        
        outputs = self.model(inputs, coord_ranges, calib, K=50, mode='test')

        dets = extract_dets_from_outputs(outputs=outputs, K=50)
        dets = dets.detach().cpu().numpy()
        # get corresponding calibs & transform tensor to numpy
        info = {key: val for key, val in info.items()}

        dets = decode_detections(
            dets = dets,
            info = info,
            calibs = calibs,
            cls_mean_size=cls_mean_size,
            threshold = self.cfg['threshold']
        )
        dets = dets[idx]
        preds = []
        class_map = {
            0: 'Pedestrian', 
            1: 'Car', 
            2: 'Cyclist'
        }

        for d in dets:
            cls = class_map[d[0]]
            tmp = [cls, 0.0, 0]
            tmp.extend(d[1:])
            preds.append(tmp)

        return preds