from lib.monoLSS.MonoLSS import MonoLSS
from lib.monoDTR.MonoDTR import MonoDTR


def build_model(cfg,mean_size,dataset_cfg):
    if cfg['type'] == 'MonoLSS':
        return MonoLSS(backbone=cfg['backbone'], neck=cfg['neck'], mean_size=mean_size)
    if cfg['type'] == 'MonoDTR':
        return MonoDTR(cfg['MonoDTR'],dataset_cfg)
    else:
        raise NotImplementedError("%s model is not supported" % cfg['type'])
