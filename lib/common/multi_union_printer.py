from lib.common.helpers.multi_processor_helper import compareAll
from lib.common.helpers.multi_processor_helper import average_lists

class Printer(object):
    def __init__(self, lss_printer, dtr_model):
        self.lss_printer = lss_printer
        self.dtr_model = dtr_model
        

    def print(self, img, calibs):
        predsDTR = self.dtr_model.print(img, calibs)
        predsLSS = self.lss_printer.print(img, calibs)
        index_pairs = compareAll(predsDTR,predsLSS)
        resultsLSS = []
        resultsDTR = []
        for dtr_idx, lss_idx in index_pairs:
            resultsLSS.append(predsLSS[lss_idx])
            resultsDTR.append(predsDTR[dtr_idx])

        averaged_list = average_lists(resultsLSS, resultsDTR)

        used_dtr = {pair[0] for pair in index_pairs}
        used_lss = {pair[1] for pair in index_pairs}

        remaining_from_list1 = [predsDTR[i] for i in range(len(predsDTR)) if i not in used_dtr]
        remaining_from_list2 = [predsLSS[i] for i in range(len(predsLSS)) if i not in used_lss]

        return averaged_list + remaining_from_list1 + remaining_from_list2
