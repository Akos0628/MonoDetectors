from lib.monoDTR.printer import Printer as PrinterDTR
from lib.monoLSS.printer import Printer as PrinterLSS
from lib.common.multi_printer import Printer as PrinterMulti
from lib.common.helpers.model_helper import build_model

def selectPrinter(cfg):
    type = cfg['type']
    if type == 'MonoLSS':
        return PrinterLSS(cfg['MonoLSS'], build_model(cfg['MonoLSS']))
    elif type == 'MonoDTR':
        return PrinterDTR(cfg['MonoDTR'], build_model(cfg['MonoDTR']))
    elif type == 'multi':
        return PrinterMulti(
            PrinterLSS(cfg['MonoLSS'], build_model(cfg['MonoLSS'])), 
            PrinterDTR(cfg['MonoDTR'], build_model(cfg['MonoDTR'])),
        )
    else:
        raise NotImplementedError("%s model is not supported" % cfg['model'])