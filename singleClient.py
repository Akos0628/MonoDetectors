import numpy as np
import glob
import cv2
import os
from lib.common.datasets.kitti_utils import get_calib_from_file

BASE_DIR = os.path.abspath('.')
SAVE_DIR = BASE_DIR + '/calibPictures'
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
CALIB_NAME = "calib.txt"
cv2.CAP_GSTREAMER
cam = cv2.VideoCapture('http://192.168.0.73:4747/mjpegfeed?1280x720')

frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = 1280
height = 384

crop = int((frame_height - height)/2)

downscale = frame_height // 1280
print(frame_height)
print(downscale)

count = 0
print(SAVE_DIR)

def calculateCalib():
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    chessWidth = 9
    chessHeight = 6
    # https://github.com/opencv/opencv/blob/4.x/doc/pattern.png
 
    objp = np.zeros((chessWidth*chessHeight,3), np.float32)
    objp[:,:2] = np.mgrid[0:chessHeight,0:chessWidth].T.reshape(-1,2)
    
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    
    images = glob.glob(SAVE_DIR + '/*.png')

    if len(images) == 0:
        print('no image found')
        return

    counter = 0
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        ret, corners = cv2.findChessboardCorners(gray, (chessHeight,chessWidth), None)
    
        if ret == True:
            #print(fname)
            counter = counter + 1
            objpoints.append(objp)
    
            corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
    
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (chessHeight,chessWidth), corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(100)
        else:
            os.remove(fname)
    cv2.destroyAllWindows()
    
    if counter == 0:
        print('No chessboard found')
        return
    print(counter)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print(mtx)


    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error
    
    print( "total error: {}".format(mean_error/len(objpoints)) )
    return mtx

P2 = None
if os.path.isfile(CALIB_NAME):
    P2 = get_calib_from_file(CALIB_NAME)


while cam.isOpened():
    ret, frame = cam.read()

    # Display the captured frame
    img = frame[crop:frame_height-crop, 0:frame_width]
    cv2.imshow('Camera', img)

    pressedKey = cv2.waitKey(1) & 0xFF
    if pressedKey == ord('q'):
        break
    elif pressedKey == ord('s'):
        name = "/frame%d.png"%count
        fullName = SAVE_DIR + name
        print(fullName)
        cv2.imwrite(fullName, img)
        count += 1
    elif pressedKey == ord('e'):
        calib = calculateCalib()
        calib = f'{calib[0,0]} 0 {calib[0,2]} 0 0 {calib[1,1]} {calib[1,2]} 0 0 0 1 0'
        print(calib)
        with open(CALIB_NAME, "w") as text_file:
            text_file.write(calib)
    elif pressedKey == ord('d'):
        if P2 == None:
            print(f'Missing {CALIB_NAME}')
        else: 
            print(f'Found {CALIB_NAME}')



# Release the capture and writer objects
cam.release()
cv2.destroyAllWindows()