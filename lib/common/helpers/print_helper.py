class PrintHelper(object):
    def __init__(self, data_loader):
        self.data_loader = data_loader

    def getPrintables(self, idx):
        dataset = self.data_loader.dataset

        img = dataset.get_image(idx)
        calibs = [dataset.get_calib(idx)]
        
        return idx, img, calibs, dataset.resolution, dataset.downsample, dataset.mean, dataset.std, dataset.cls_mean_size
