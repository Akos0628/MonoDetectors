from torch.utils.data import DataLoader
from lib.common.datasets.kitti import KITTI

def build_dataset(cfg, mode):
    # --------------  build kitti dataset ----------------
    if cfg['type'] == 'kitti':
        if mode in ['train', 'val']:
            return KITTI(root_dir=cfg['root_dir'], split='train', cfg=cfg)
        else: 
            return KITTI(root_dir=cfg['root_dir'], split='test', cfg=cfg)
    else:
        raise NotImplementedError("%s dataset is not supported" % cfg['type'])

