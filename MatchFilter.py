import cv2
import matplotlib.image
import matplotlib.pyplot as plt
import numpy as np

class MatchFilter():

    def __init__(self):

        #self.target = matplotlib.image.imread('target.png')
        #self.target = matplotlib.image.imread('target_hblur.png')
        #self.target = matplotlib.image.imread('righthand.png')
        self.target = matplotlib.image.imread('target4.png')

        self.target = cv2.cvtColor(self.target, cv2.COLOR_RGB2GRAY)
        self.target = (255*self.target).astype(np.uint8)
        self.target = cv2.resize(self.target, (0,0), fx=0.1, fy=0.1)
        self.target = cv2.equalizeHist(self.target)
        ret, self.target = cv2.threshold(self.target,100,255, cv2.THRESH_BINARY)

        self.target = cv2.Laplacian(self.target, cv2.CV_8U)
        self.target = self.target.astype(np.float32)

        #plt.imshow(self.target)
        #plt.show()
        print(self.target.shape)

        #self.rh1_ft = np.fft.fft2(self.righthand1)
        #transpose for correlation

    def correlate(self, image):
        return cv2.matchTemplate(image, self.target, cv2.TM_CCORR)

    def getTemplate(self):
        return self.target



        

