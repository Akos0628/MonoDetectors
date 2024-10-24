import numpy as np
import fire
import cv2
import os
from timeit import default_timer as timer
from datetime import timedelta
from multiprocessing import Pool, Queue
import time
from copy import deepcopy

from lib.common.helpers.detection_helper import predArrayToResult
from lib.common.helpers.visualization_helper import visualizationCv
from lib.common.datasets.kitti_utils import get_P2_from_file
from lib.common.datasets.kitti_utils import get_calibs_from_P2
from lib.common.datasets.kitti_utils import convertP2StringToCalib
from lib.common.helpers.multi_processor_helper import compareAll
from lib.common.helpers.multi_processor_helper import average_lists

import grpc
from services import detector_pb2
from services import detector_pb2_grpc

CALIB_NAME = "calib.txt"

P2 = None
if os.path.isfile(CALIB_NAME):
    print('calib file exists')
    P2 = get_P2_from_file(CALIB_NAME)
    print(P2)

def getDetect(img_brg, calib, stub, width, height, treshold):
    img = cv2.cvtColor(img_brg, cv2.COLOR_BGR2RGB)
    success, encoded_image = cv2.imencode('.png', img)
    img_byte_arr = encoded_image.tobytes()

    response = stub.Detect(
        detector_pb2.DetectRequest(
            width=width,
            height=height,
            data=img_byte_arr,
            calib=calib,
            treshold=treshold
        )
    )
    calibs = get_calibs_from_P2(calib)
    result = predArrayToResult(response.detections)
    return calibs, result

def detct_dtr(serverAddress, img_brg, width, height, calib, treshold):
    with grpc.insecure_channel(serverAddress) as channel:
        stub = detector_pb2_grpc.DetectorStub(channel)
        start_time = timer()
        calibs, result = getDetect(img_brg, calib, stub, width, height, treshold)
        end_time = timer()
        #print(timedelta(seconds=end_time - start_time))

        return result, img_brg

def detct_lss(serverAddress, img_brg, width, height, calib, treshold):
    with grpc.insecure_channel(serverAddress) as channel:
        stub = detector_pb2_grpc.DetectorStub(channel)
        start_time = timer()
        calibs, result = getDetect(img_brg, calib, stub, width, height, treshold)
        end_time = timer()
        #print(timedelta(seconds=end_time - start_time))

        return result, img_brg

def run(
    captureDevice:str='http://192.168.0.73:4747/mjpegfeed?1280x720',
    serverAddress_dtr:str='localhost:50051',
    serverAddress_lss:str='localhost:50052',
    treshold_dtr:float=0.3,
    treshold_lss:float=0.1
):
    global P2
    if P2 == None:
        return
    calib = convertP2StringToCalib(P2)
    calibs = get_calibs_from_P2(calib)
    pool = Pool(2)

    cam = cv2.VideoCapture(captureDevice)

    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(frame_width)
    print(frame_height)
    width = 1280
    height = 384

    crop_height = int((frame_height - height)/2)
    crop_width = int((frame_width - width)/2)

    future_dtr = None
    future_lss = None
    while cam.isOpened():
        ret, frame = cam.read()
        if ret:
            if future_dtr == None and future_lss == None:
                img_brg = frame[crop_height:frame_height-crop_height, crop_width:frame_width-crop_width]
                future_dtr = pool.apply_async(detct_dtr, args=(serverAddress_dtr, img_brg, width, height, calib, treshold_dtr,))
                future_lss = pool.apply_async(detct_lss, args=(serverAddress_lss, img_brg, width, height, calib, treshold_lss,))
            else:
                if future_dtr.ready() and future_lss.ready():
                    predsDTR, img_res1 = future_dtr.get()
                    predsLSS, img_res2 = future_lss.get()
                    assert(img_res1, img_res1)
                    
                    same = compareAll(predsDTR,predsLSS)
                    resultsLSS = []
                    resultsDTR = []
                    for dtr_idx, lss_idx in same:
                        resultsLSS.append(predsLSS[lss_idx])
                        resultsDTR.append(predsDTR[dtr_idx])

                    result = average_lists(resultsLSS, resultsDTR)
                    visualizationCv(img_res1, calibs, result)
                    future_dtr = None
                    future_lss = None

                    cv2.imshow('Camera', img_res1)
                    cv2.waitKey(1)

        
        else:
            break
        
    future_dtr.get()
    future_lss.get()
    print("program ended")
    
    # Release the capture and writer objects
    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    fire.Fire(run)
