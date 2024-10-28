from lib.common.helpers.multi_processor_helper import compareAll
from lib.common.helpers.multi_processor_helper import average_lists

class Printer(object):
    def __init__(self, lss_printer, dtr_model):
        self.lss_printer = lss_printer
        self.dtr_model = dtr_model
        

    def print(self, img, calibs):
        predsDTR = self.dtr_model.print(img, calibs)
        predsLSS = self.lss_printer.print(img, calibs)
        same = compareAll(predsDTR,predsLSS)
        resultsLSS = []
        resultsDTR = []
        for dtr_idx, lss_idx in same:
            resultsLSS.append(predsLSS[lss_idx])
            resultsDTR.append(predsDTR[dtr_idx])

        result = average_lists(resultsLSS, resultsDTR)

        return result
