import numpy as np
import matplotlib.pyplot as plt
import cv2

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