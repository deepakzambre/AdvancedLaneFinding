import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#from moviepy.editor import VideoFileClip

from LaneFinder import LaneFinder
from PerspectiveTransformer import PerspectiveTransformer
from Thresholder import Thresholder
from CameraHelper import CameraHelper

def TestUndistortion():
    
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    img = mpimg.imread('./camera_cal/calibration1.jpg')
    ax1.imshow(img)
    ax2.imshow(cameraHelper.Undistort(img))
    ax1.set_title('Original Image', fontsize=50)
    ax2.set_title('Undistorted Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

def TestThresholder():
    
    images = glob.glob('./test_images/*.jpg')
    
    for idx, fname in enumerate(images):
        
        img = mpimg.imread(fname)
        undistorted = cameraHelper.Undistort(img)
        
        fig, (axs1, axs2, axs3, axs4) = plt.subplots(1, 4, figsize=(24, 9))
        axs1.imshow(img)
        axs2.imshow(thresholder.ApplySobelThreshold(undistorted), cmap = 'gray')
        axs3.imshow(thresholder.ApplyColorThreshold(undistorted), cmap = 'gray')
        axs4.imshow(thresholder.ApplySobelAndColorThreshold(undistorted), cmap = 'gray')
        
        axs1.set_title(fname)
        axs2.set_title('Gradient threshold')
        axs3.set_title('HLS threshold')
        axs4.set_title('Combined')

def TestPerspectiveTransformation(axs, fpath):

    images = glob.glob(fpath)
    
    for idx, fname in enumerate(images):
        
        img = mpimg.imread(fname)
        undistorted = cameraHelper.Undistort(img)
        
        poly = np.zeros_like(img)
        cv2.fillPoly(poly, np.int32([lanePoints]), (0,255, 0))
        undistorted = cv2.addWeighted(undistorted, 1, poly, 0.3, 0)

        axs[int(idx/2)][(idx % 2) * 2].imshow(undistorted)
        axs[int(idx/2)][(idx % 2) * 2 + 1].imshow(perspectiveTransformer.ToBirdsEyeView(undistorted), cmap = 'gray')

def TestLaneFinder(fname, cameraHelper, thresholder, perspectiveTransformer, laneFinder):
    
    img = mpimg.imread(fname)
    undistorted = cameraHelper.Undistort(img)
    combined = thresholder.ApplySobelAndColorThreshold(undistorted)
    birdsEye = perspectiveTransformer.ToBirdsEyeView(combined)
    laneFinder.__init__()
    laneFinder.FindLane(birdsEye)
    laneFinder.Visualize(fname, img, birdsEye)

def Main():

    cameraHelper = CameraHelper('./camera_cal/')
    thresholder = Thresholder()
    perspectiveTransformer = PerspectiveTransformer(cameraHelper)
    laneFinder = LaneFinder()

    #TestUndistortion()

    #TestThresholder()

    #fig, axs = plt.subplots(4, 4, figsize=(24, 9))
    #TestPerspectiveTransformation(axs, './test_images/*.jpg')

    images = glob.glob('./test_images/*.jpg')
    for idx, fname in enumerate(images):
        if idx is not 4: # or 7
            continue

        TestLaneFinder(fname, cameraHelper, thresholder, perspectiveTransformer, laneFinder)

    return

if __name__ == '__main__':
    Main()
