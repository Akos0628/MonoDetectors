class PrintHelper(object):
    def __init__(self, dataset):
        self.dataset = dataset

    def getPrintables(self, idx):
        img = self.dataset.get_image(idx)
        calibs = [self.dataset.get_calib(idx)]
        
        return img, calibs
    
    def getRealPrintables(self, idx):
        img = self.dataset.get_image(idx)
        calib = self.dataset.get_calib_P2(idx)
        
        return img, calib
