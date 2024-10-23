import numpy as np
import fire
import cv2
import os
from timeit import default_timer as timer
from datetime import timedelta

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
    #print(f"Result: {result}")
    return calibs, result

def run(
    captureDevice:str='http://192.168.0.73:4747/mjpegfeed?1280x720',
    serverAddress:str='localhost:50051',
    treshold:float=0.02
):
    cam = cv2.VideoCapture(captureDevice)
    cam.set(cv2.CAP_PROP_FPS, 5)

    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(frame_width)
    print(frame_height)
    width = 1280
    height = 384

    crop_height = int((frame_height - height)/2)
    crop_width = int((frame_width - width)/2)

    global P2
    frame_rate = 15
    prev = 0
    
    while cam.isOpened():
        time_elapsed = timer() - prev
        ret, frame = cam.read()


        # Display the captured frame
        img_brg = frame[crop_height:frame_height-crop_height, crop_width:frame_width-crop_width]

        pressedKey = cv2.waitKey(1) & 0xFF
        if pressedKey == ord('q'):
            break
        else:
            if time_elapsed > 1./frame_rate:
                prev = timer()
                if P2 == None:
                    return
                else: 
                    with grpc.insecure_channel(serverAddress) as channel:
                        stub = detector_pb2_grpc.DetectorStub(channel)
                        start_time = timer()
                        calibs, result = getDetect(img_brg, convertP2StringToCalib(P2), stub, width, height, treshold)
                        end_time = timer()
                        print(timedelta(seconds=end_time - start_time))
                        visualizationCv(img_brg, calibs, result)

                        #img = cv2.cvtColor(img_brg, cv2.COLOR_BGR2RGB)
                        #visualization(img, calibs, result, True, False, False)

                        cv2.imshow('Camera', img_brg)

    # Release the capture and writer objects
    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    fire.Fire(run)
