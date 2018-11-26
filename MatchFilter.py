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

class TargetFinder():

    def __init__(self):
        self.xpos = None
        self.ypos = None

    def findTarget(self, source):

        #Generating positions arrays
        if self.xpos is None:
            self.generatePositions(source)

        height = source.shape[0]
        width = source.shape[1]

        #Dimensional sum and convert to probabilities
        verticalSum = np.sum(source,axis=1)
        horizontalSum = np.sum(source,axis=0)

        v0norm = np.sum(verticalSum)
        h0norm = np.sum(horizontalSum)

        if (v0norm > 0): verticalSum = verticalSum/v0norm
        if (h0norm > 0): horizontalSum = horizontalSum/h0norm

        #Mean position
        mean_y = np.sum(verticalSum*self.ypos)
        mean_x = np.sum(horizontalSum*self.xpos)

        #Max position
        in_val, max_val, min_loc, max_loc = cv2.minMaxLoc(source)

        #np.savetxt("foo.csv", verticalSum, delimiter=",")

        x = max_loc[0]
        y = max_loc[1]
        
        return x,y


    def generatePositions(self, source):
        height = source.shape[0]
        width = source.shape[1]

        #starting at 1 to avoid the times zero problem
        self.xpos = np.linspace(0,width-1,width)
        self.ypos = np.linspace(0,height-1,height)


        

