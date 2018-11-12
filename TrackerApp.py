from tkinter import *
from Camstream import Camstream
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle, Circle
import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2
from MatchFilter import MatchFilter
import numpy as np
import time
from threading import Thread
from TargetFinder import TargetFinder
from Kalman import IMM
from math import sin, cos

class TrackerApp(Frame):

    def __init__(self, mainFrame, filter):
        super().__init__(mainFrame)

        self.pack(side = LEFT)

        #Initialize output graph
        self.outFigure = Figure(figsize=(8,6), dpi=100)
        self.outPlot = self.outFigure.add_axes([0,0,1,1])
        self.outCanvas = FigureCanvasTkAgg(self.outFigure, self)
        self.outCanvas.get_tk_widget().pack(side=TOP,fill=BOTH,expand=False)

        self.outFigure.get_axes()[0].get_xaxis().set_visible(False)
        self.outFigure.get_axes()[0].get_yaxis().set_visible(False) 
        self.outPlot.axis('off')

        self.infoLabel = self.outPlot.text(0.05,0.95,"Initializaing...")

        #initialize matplotlib figure for image
        self.imFigure = Figure(figsize=(8,2), dpi=100)
        self.imPlot = self.imFigure.add_subplot(121)
        self.imPlot2 = self.imFigure.add_subplot(122)
        self.imCanvas = FigureCanvasTkAgg(self.imFigure, self)
        self.imCanvas.get_tk_widget().pack(side=BOTTOM, fill=BOTH, expand=False)

        #detection settings
        self.frames = 0
        self.dtreshold = 0
        self.voidPrior = None

        #set filter (particle, kalman)
        self.filter = filter

        #Initialize matchfilter
        self.matchfilter = MatchFilter()

        #Initialize targetFinder
        self.targetFinder = TargetFinder()

        #initialize camera
        self.running = True
        self.camstream = Camstream(self)

        #initialize draw thread
        self.TrackerAppThread = TrackerAppThread(self.imPlot, self.imPlot2, self.imCanvas)
        


    def updateImage(self, img, delta):

        #converting image to grayscale (image is uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_backup = img

        #histogram equalization
        img = cv2.equalizeHist(img)

        #remove noise
        img = cv2.GaussianBlur(img, (3,3), 1)
        
        #Laplacian edge detection and thresholding tozero
        img = cv2.Laplacian(img,cv2.CV_8U)
        res, img = cv2.threshold(img, np.max(img)/3, np.max(img),cv2.THRESH_TOZERO)
        img = img.astype(np.float32)


        #plotting matching map
        correlated = self.matchfilter.correlate(img)
        in_val, max_val, min_loc, max_loc = cv2.minMaxLoc(correlated)
        #rect2 = Rectangle((x,y),5,5,linewidth=1,edgecolor='r',facecolor='none')

        if self.frames<10:
            #First frames serves as initialization of the detection threshold
            self.frames+=1
            self.dtreshold += max_val/10

            if self.voidPrior is None:
                self.voidPrior = correlated
            else:
                self.voidPrior += correlated

            if self.frames == 10:
                #self.voidPrior /= 10
                #minv, maxv, minloc, maxloc = cv2.minMaxLoc(source)
                pass

        else:

            #Thresholding the detection map
            res, thr = cv2.threshold(correlated,self.dtreshold*0.9,max_val,cv2.THRESH_TOZERO)

            #Finding the target
            x,y = self.targetFinder.findTarget(thr)
            rect2 = Rectangle((x,y),5,5,linewidth=1,edgecolor='r',facecolor='none')
            self.imPlot2.add_patch(rect2)


            #Apply filtering
            #s = self.filter.compute(x,y, delta)

            #rect3 = Rectangle((s[0][0], s[1][0]),5,5,linewidth=1,edgecolor='b',facecolor='none')
            #nx = s[0][0]
            #ny = s[1][0]
            #r = s[4][0]
            #theta = s[2][0]
            #circle = Circle((nx - r*cos(theta), ny - r*sin(theta)), radius = r, linewidth=1, edgecolor='b', facecolor='none')
            #self.imPlot2.add_patch(rect3)
            #self.imPlot2.add_patch(circle)

            #Update plots
            self.TrackerAppThread.setImages(img_backup, thr)


    def closeCam(self):
        self.camstream.closeStream()





class TrackerAppThread(Thread):

    def __init__(self, sourcePlot, outputPlot, canvas):
        self.sourcePlot = sourcePlot
        self.outputPlot = outputPlot
        self.isRunning = False
        self.canvas = canvas

        self.sourceImage = None
        self.outImage = None

        self.startThread()

    def startThread(self):

        #Initializing thread
        Thread.__init__(self)

        self.framesDrawn = 0
        self.lastTime = 0

        #Close thread when exiting program
        self.setDaemon(True)

        #Starting Thread
        self.isRunning = True
        self.lastTime = time.time()
        self.start()

    def setImages(self, source, out):
        self.sourceImage = source
        self.outImage = out
        self.framesDrawn += 1


    def run(self):
        while (self.isRunning):
            if (self.sourceImage is not None):
                self.sourcePlot.clear()
                self.sourcePlot.imshow(self.sourceImage, interpolation=None)
                self.sourcePlot.axis('off')

            if (self.outImage is not None):
                self.outputPlot.clear()
                self.outputPlot.imshow(self.outImage, interpolation=None)
                self.outputPlot.axis('off')

            self.canvas.draw()

            now = time.time()
            fps = 1/(now-self.lastTime)
            #print("PROCESSED FPS : {:d}     GUI FPS : {:d}".format(int(fps*self.framesDrawn), int(fps)))
            self.framesDrawn = 0
            self.lastTime = now

            time.sleep(0.001)
            

            
                
