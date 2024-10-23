import numpy as np
import glob
import cv2
import os

def calculateCalib(saveDir):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    chessWidth = 9
    chessHeight = 6
    # https://github.com/opencv/opencv/blob/4.x/doc/pattern.png
 
    objp = np.zeros((chessWidth*chessHeight,3), np.float32)
    objp[:,:2] = np.mgrid[0:chessHeight,0:chessWidth].T.reshape(-1,2)
    
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    
    images = glob.glob(saveDir + '/*.png')

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
