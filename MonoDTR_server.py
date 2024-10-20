import os

import grpc
from concurrent import futures
from services import detector_pb2
from services import detector_pb2_grpc
import yaml
import io
import torch
import numpy as np
from PIL import Image
from timeit import default_timer as timer
from datetime import timedelta

from lib.common.helpers.model_helper import build_model
from lib.common.helpers.detection_helper import stringFromLine
from lib.common.datasets.kitti_utils import get_calibs_from_P2

mode = 'test' # test, eval, train
config = 'configs/dtr.yaml'

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# load cfg
assert (os.path.exists(config))
cfg = yaml.load(open(config, 'r'), Loader=yaml.Loader)

from lib.monoDTR.printer import Printer as PrinterDTR
MonoDTR = PrinterDTR(cfg['MonoDTR'], build_model(cfg['MonoDTR']))


class DetectorServicer(detector_pb2_grpc.DetectorServicer):
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

        preds = MonoDTR.print(img, calibs, treshold)
        
        realPreds = []
        for line in preds:
            newLine = stringFromLine(line)
            realPreds.append(newLine)
        #print(realPreds)
        end_time = timer()
        print(timedelta(seconds=end_time - start_time))
        return detector_pb2.DetectResponse(detections=realPreds)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    detector_pb2_grpc.add_DetectorServicer_to_server(DetectorServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()