from lib.monoDTR.printer import Printer as PrinterDTR
from lib.monoLSS.printer import Printer as PrinterLSS
from lib.common.multi_intersection_printer import Printer as PrinterMultiIntersection
from lib.common.multi_union_printer import Printer as PrinterMultiUnion
from lib.common.weighted_multi_printer import Printer as PrinterMultiWeighted
from lib.common.helpers.model_helper import build_model

def selectPrinter(cfg):
    type = cfg['type']
    if type == 'MonoLSS':
        return PrinterLSS(cfg['MonoLSS'], build_model(cfg['MonoLSS']))
    elif type == 'MonoDTR':
        return PrinterDTR(cfg['MonoDTR'], build_model(cfg['MonoDTR']))
    elif type == 'multi-intersect':
        return PrinterMultiIntersection(
            PrinterLSS(cfg['MonoLSS'], build_model(cfg['MonoLSS'])), 
            PrinterDTR(cfg['MonoDTR'], build_model(cfg['MonoDTR'])),
        )
    elif type == 'multi-union':
        return PrinterMultiUnion(
            PrinterLSS(cfg['MonoLSS'], build_model(cfg['MonoLSS'])), 
            PrinterDTR(cfg['MonoDTR'], build_model(cfg['MonoDTR'])),
        )
    elif type == 'multi-weighted':
        return PrinterMultiWeighted(
            PrinterLSS(cfg['MonoLSS'], build_model(cfg['MonoLSS'])), 
            PrinterDTR(cfg['MonoDTR'], build_model(cfg['MonoDTR'])),
        )
    else:
        raise NotImplementedError("%s model is not supported" % cfg['model'])