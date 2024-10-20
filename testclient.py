import grpc
import os

from services import detector_pb2
from services import detector_pb2_grpc

from timeit import default_timer as timer
from datetime import timedelta
import io
import yaml

from lib.common.helpers.dataloader_helper import build_dataloader
from lib.common.helpers.visualization_helper import visualization
from lib.common.helpers.detection_helper import predArrayToResult
from lib.common.helpers.print_helper import PrintHelper
from lib.common.datasets.kitti_utils import get_calibs_from_P2

mode = 'test' # test, eval, train
config = 'configs/kitti-data.yaml'

# load cfg
assert (os.path.exists(config))
cfg = yaml.load(open(config, 'r'), Loader=yaml.Loader)
  
#  build dataloader
dataset = build_dataloader(cfg['dataset'], mode)
printHelper = PrintHelper(dataset)

def run(img, calib, stub):
    calibs = get_calibs_from_P2(calib)

    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    response = stub.Detect(
        detector_pb2.DetectRequest(
            width=img.width,
            height=img.height,
            data=img_byte_arr,
            calib=calib,
            treshold=0.01
        )
    )
    result = predArrayToResult(response.detections)
    #print(f"Result: {result}")
    return img, calibs, result


if __name__ == '__main__':
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = detector_pb2_grpc.DetectorStub(channel)
        while True:
            num1 = int(input("Please input num1: "))
            img, calib = printHelper.getRealPrintables(num1)
            start_time = timer()
            img, calibs, result = run(img, calib, stub)
            end_time = timer()
            print(timedelta(seconds=end_time - start_time))
            visualization(img, calibs, result, True, False, False)