import os
import torch.utils.data as data
from PIL import Image

from lib.common.datasets.kitti_utils import Calibration

class KITTI(data.Dataset):
    def __init__(self, root_dir, split, cfg):
        # data split loading
        assert split in ['train', 'test']

        # path configuration
        self.data_dir = os.path.join(root_dir, cfg['data_dir'], 'testing' if split == 'test' else 'training')
        self.image_dir = os.path.join(self.data_dir, 'image_2')
        self.calib_dir = os.path.join(self.data_dir, 'calib')

    def get_image(self, idx):
        img_file = os.path.join(self.image_dir, '%06d.png' % idx)
        assert os.path.exists(img_file)
        return Image.open(img_file)    # (H, W, 3) RGB mode

    def get_calib(self, idx):
        calib_file = os.path.join(self.calib_dir, '%06d.txt' % idx)
        assert os.path.exists(calib_file)
        return Calibration(calib_file)
    