import os
import io
import yaml
import grpc
import fire
import torch
from PIL import Image
from concurrent import futures
from services import detector_pb2
from services import detector_pb2_grpc
from timeit import default_timer as timer
from datetime import timedelta

from lib.common.helpers.detection_helper import stringFromLine
from lib.common.helpers.print_selector import selectPrinter
from lib.common.datasets.kitti_utils import get_calibs_from_P2

# cuda available
assert (torch.cuda.is_available())
printer = None

class DetectorServicer(detector_pb2_grpc.DetectorServicer):
    def __init__(self):
        assert(printer != None)

    def Detect(self, request, context):
        start_time = timer()

        width = request.width
        height = request.height
        calib = request.calib
        treshold = request.treshold
        data = request.data

        stream = io.BytesIO(data)
        img = Image.open(stream)
        calibs = get_calibs_from_P2(calib)

        preds = printer.print(img, calibs, treshold)
        
        realPreds = []
        for line in preds:
            newLine = stringFromLine(line)
            realPreds.append(newLine)
        #print(realPreds)
        end_time = timer()
        print(timedelta(seconds=end_time - start_time))
        return detector_pb2.DetectResponse(detections=realPreds)

def serve(
    config:str="configs/lss.yaml",
    port:str='50051'
):
    print(f'config is: {config}')
    print(f'port is: {port}')
    # load cfg
    assert (os.path.exists(config))
    cfg = yaml.load(open(config, 'r'), Loader=yaml.Loader)

    global printer
    printer = selectPrinter(cfg)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    detector_pb2_grpc.add_DetectorServicer_to_server(DetectorServicer(), server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    fire.Fire(serve)