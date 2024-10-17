from lib.monoLSS.MonoLSS import MonoLSS
from lib.monoDTR.MonoDTR import MonoDTR


def build_model(cfg):
    if cfg['type'] == 'MonoLSS':
        return MonoLSS(backbone=cfg['backbone'], neck=cfg['neck'], mean_size=cfg['cls_mean_size'])
    if cfg['type'] == 'MonoDTR':
        return MonoDTR(cfg)
    else:
        raise NotImplementedError("%s model is not supported" % cfg['type'])
