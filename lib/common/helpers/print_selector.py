
from lib.monoDTR.printer import Printer as PrinterDTR
from lib.monoLSS.printer import Printer as PrinterLSS
from lib.common.helpers.model_helper import build_model

def selectPrinter(cfg):
    type = cfg['type']
    if type == 'MonoLSS':
        return PrinterLSS(cfg, build_model(cfg))
    elif type == 'MonoDTR':
        return PrinterDTR(cfg, build_model(cfg))
    else:
        raise NotImplementedError("%s model is not supported" % cfg['model'])