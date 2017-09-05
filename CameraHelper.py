import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#%matplotlib inline
#%matplotlib qt

class CameraHelper:

    def __init__(self, calibration_image_dir):

        objp = np.zeros((6*9,3), np.float32)
        objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)
        
        objpoints = []
        imgpoints = []
        
        images = glob.glob(calibration_image_dir + 'calibration*.jpg')
        img_shape = ()
        
        for idx, fname in enumerate(images):
            
            img = mpimg.imread(fname)
            img_shape = (img.shape[1], img.shape[0])
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)
                
        self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, img_shape, None, None)

    def Undistort(self, img):
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
