import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class Thresholder:
    
    def __init__(self,
                 sobel_kernel = 15,
                 threshx = (20, 100),
                 threshy = (20, 100),
                 threshmag = (40, 120),
                 threshdir = (0.77, 1.6),
                 threshhls=(90, 255)):
        
        self.sobel_kernel = sobel_kernel
        self.threshx = threshx
        self.threshy = threshy
        self.threshmag = threshmag
        self.threshdir = threshdir
        self.threshhls = threshhls
    
    def ApplySobelThreshold(self, img):
        
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self.sobel_kernel)
        
        absx = np.absolute(sobelx)
        x = np.uint8(255 * absx / np.max(absx))

        absy = np.absolute(sobely)
        y = np.uint8(255 * absy / np.max(absy))
        
        mag = np.sqrt(sobelx**2 + sobely**2)
        mag = (mag * 255 / np.max(mag)).astype(np.uint8) 

        dir = np.arctan2(sobely, sobelx)
        
        binary = np.zeros_like(gray)
        binary[(((x >= self.threshx[0]) & (x <= self.threshx[1])) & ((y >= self.threshy[0]) & (y <= self.threshy[1])))
              | (((mag >= self.threshmag[0]) & (mag <= self.threshmag[1])) & ((dir >= self.threshdir[0]) & (dir <= self.threshdir[1])))
              ] = 1
        
        return binary
    
    def ApplyColorThreshold(self, img):

        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        s_channel = hls[:,:,2]
        binary_output = np.zeros_like(s_channel)
        binary_output[(s_channel > self.threshhls[0]) & (s_channel <= self.threshhls[1])] = 1

        return binary_output
    
    def ApplySobelAndColorThreshold(self, undistorted):
        
        sobeled = self.ApplySobelThreshold(undistorted)
        colored = self.ApplyColorThreshold(undistorted)
        
        combined = np.zeros_like(colored)
        combined[(sobeled == 1)] = 60
        combined[(colored == 1)] += 180
        
        return combined
