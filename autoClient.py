import numpy as np
import fire
import cv2
import os
from timeit import default_timer as timer
from datetime import timedelta
from multiprocessing import Pool, Queue
import time

from lib.common.helpers.calib_calculator import calculateCalib
from lib.common.helpers.detection_helper import predArrayToResult
from lib.common.helpers.visualization_helper import visualizationCv
from lib.common.helpers.visualization_helper import visualization
from lib.common.datasets.kitti_utils import get_P2_from_file
from lib.common.datasets.kitti_utils import get_calibs_from_P2
from lib.common.datasets.kitti_utils import convertP2StringToCalib

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

def init_pool(d_b, p_b):
    global detection_buffer
    global processed_buffer
    detection_buffer = d_b
    processed_buffer = p_b


def detect_object(img_brg):
    detection_buffer.put(img_brg)

def detct(serverAddress, width, height, treshold):
    global P2
    if P2 == None:
        return
    while True:
        img_brg = detection_buffer.get()
        if img_brg is not None:
            with grpc.insecure_channel(serverAddress) as channel:
                stub = detector_pb2_grpc.DetectorStub(channel)
                start_time = timer()
                calibs, result = getDetect(img_brg, convertP2StringToCalib(P2), stub, width, height, treshold)
                end_time = timer()
                #print(timedelta(seconds=end_time - start_time))
                visualizationCv(img_brg, calibs, result)

                processed_buffer.put({ 'img_brg': img_brg, 'calibs': calibs, 'result': result})
        else:
            break
    return

def show():
    while True:
        data = processed_buffer.get()
        if data is not None:
            img_brg = data['img_brg']
            calibs = data['calibs']
            result = data['result']
            visualizationCv(img_brg, calibs, result)

            cv2.imshow('Camera', img_brg)
        else:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    return

def run(
    captureDevice:str='http://192.168.0.73:4747/mjpegfeed?1280x720',
    serverAddress:str='localhost:50051',
    treshold:float=0.02
):
    detection_buffer = Queue()
    processed_buffer = Queue()
    pool = Pool(3, initializer=init_pool, initargs=(detection_buffer,processed_buffer))

    cam = cv2.VideoCapture(captureDevice)

    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(frame_width)
    print(frame_height)
    width = 1280
    height = 384

    crop_height = int((frame_height - height)/2)
    crop_width = int((frame_width - width)/2)
    detct_future = pool.apply_async(detct, args=(serverAddress, width, height, treshold,))
    show_future = pool.apply_async(show)

    futures = []
    while cam.isOpened():
        ret, frame = cam.read()
        if ret:
            if detection_buffer.qsize() < 2:
                img_brg = frame[crop_height:frame_height-crop_height, crop_width:frame_width-crop_width]
                f = pool.apply_async(detect_object, args=(img_brg,))
                futures.append(f)
            else:
                detection_buffer.get()
        else:
            break
        # Close when window closed
        if show_future.ready():
            break
        
    for f in futures:
        f.get()
    detection_buffer.put(None)
    processed_buffer.put(None)
    show_future.get()
    detct_future.get()
    print("program ended")
    
    # Release the capture and writer objects
    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    fire.Fire(run)
