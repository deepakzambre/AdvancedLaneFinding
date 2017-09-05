import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#12 pixel increase
#lanePoints = np.array([[592, 444], [700, 444], [1276, 691], [125, 691]], np.float32)
lanePoints = np.array([[595, 444], [695, 444], [1276, 719], [125, 719]], np.float32)

class PerspectiveTransformer():
    
    def __init__(self, cameraHelper):
        
        img = mpimg.imread('./test_images/straight_lines2.jpg')
        img = cameraHelper.Undistort(img)

        dst = np.float32([[100, 100],
        [img.shape[1] - 100, 100],
        [img.shape[1] - 100, img.shape[0] - 100],
        [100, img.shape[0] - 100]])
        
        self.M = cv2.getPerspectiveTransform(lanePoints, dst)
        self.InvM = cv2.getPerspectiveTransform(dst, lanePoints)
        
    def ToBirdsEyeView(self, img):
   
        return cv2.warpPerspective(img, self.M, (img.shape[1], img.shape[0]), flags=cv2.INTER_NEAREST)

    def ToDashCamView(self, img):
        
        return cv2.warpPerspective(img, self.InvM, (img.shape[1], img.shape[0]), flags=cv2.INTER_NEAREST)

