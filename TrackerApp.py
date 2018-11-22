from tkinter import *
from Camstream import Camstream
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle, Circle, ConnectionPatch, Arrow
import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2
from MatchFilter import MatchFilter
import numpy as np
import time
from threading import Thread
from TargetFinder import TargetFinder
from Kalman import KalmanIMM
from math import sin, cos, sqrt, atan2

colors = ['b', 'g', 'orange']

class TrackerApp(Frame):

    def __init__(self, mainFrame, filter):
        super().__init__(mainFrame)

        self.pack(side = LEFT)

        #Initialize output graph
        self.outFigure = Figure(figsize=(8,6), dpi=100)
        self.outPlot = self.outFigure.add_axes([0.05,0.05,0.9,0.9])
        self.outCanvas = FigureCanvasTkAgg(self.outFigure, self)
        self.outCanvas.get_tk_widget().pack(side=TOP,fill=BOTH,expand=False)

        #self.outFigure.get_axes()[0].get_xaxis().set_visible(False)
        #self.outFigure.get_axes()[0].get_yaxis().set_visible(False)
        self.outFigure.get_axes()[0].set_xlim(150,0)
        self.outFigure.get_axes()[0].set_ylim(150,0)
        #self.outPlot.axis('off')

        self.outputPlotGraphics = OutputPlotGraphics(self.outPlot)

        self.infoLabel = self.outPlot.text(10,10,"Label")

        self.infoLabelText = "Initializing..."

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
        self.objectlostcounter = 0

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
        self.trackerAppThread = TrackerAppThread(self, self.imPlot, self.imPlot2, self.imCanvas, self.outCanvas, self.outPlot, self.infoLabel)
        


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
            #rect2 = Rectangle((x,y),5,5,linewidth=1,edgecolor='r',facecolor='none')
            #self.imPlot2.add_patch(rect2)


            #Apply filtering
            s, p = self.filter.compute(x,y, delta)

            for i, state in enumerate(s):
                #print(state[0][0], state[1][0])
                r = Rectangle((state[0][0], state[1][0]), 5, 5, linewidth=1, edgecolor=colors[i], facecolor='none')
                self.imPlot2.add_patch(r)
                self.imPlot2.text(state[0][0], state[1][0], str(i))

            self.trackerAppThread.setInfoText(str(np.argmax(p)) + str(p))

            if np.argmax(p)==0:
                self.objectlostcounter += 1
                if self.objectlostcounter >= 50:
                    self.objectlostcounter = 0
                    self.filter.resetState(x, y)
                    print("filter reset")
            else:
                self.objectlostcounter = 0
            
            if np.argmax(p)==2:
                print(s[2][4])
            

            #Update plots
            self.trackerAppThread.setImages(img_backup, thr)

            #measurement
            self.outputPlotGraphics.setMeasurement(x,y)

            #state
            n = np.argmax(p)
            self.outputPlotGraphics.setModel(np.argmax(p))

            if n != 0:
                self.outputPlotGraphics.setSpeed(s[n][0], s[n][1], s[n][2], s[n][3])
                self.outputPlotGraphics.setState(s[n][0], s[n][1])

                if n==1:
                    self.outputPlotGraphics.setLine(s[n][0], s[n][1])
                if n==2:
                    R = sqrt(s[n][2]**2 + s[n][3]**2)/s[n][4]
                    theta = atan2(s[n][3], s[n][2])
                    x0 = s[n][0] + R*cos(theta)
                    y0 = s[n][1] + R*sin(theta)
                    #print(s[n][4])
                    self.outputPlotGraphics.setCircle(x0, y0, R)
            




    def closeCam(self):
        self.camstream.closeStream()





class TrackerAppThread(Thread):

    def __init__(self, camview, sourcePlot, outputPlot, canvas, outCanvas, outPlot, label):
        self.sourcePlot = sourcePlot
        self.outputPlot = outputPlot
        self.isRunning = False
        self.canvas = canvas
        self.label = label
        self.outCanvas = outCanvas
        self.outPlot = outPlot
        self.camview = camview
        
        self.infoText = "Init..."

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

    def setInfoText(self, txt):
        self.infoText = txt


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
            
            #self.outPlot.clear()
            self.label.set_text(self.infoText)
            self.outCanvas.draw()

            now = time.time()
            fps = 1/(now-self.lastTime)
            print("PROCESSED FPS : {:d}     GUI FPS : {:d}".format(int(fps*self.framesDrawn), int(fps)))
            self.framesDrawn = 0
            self.lastTime = now

            time.sleep(0.001)
            


class OutputPlotGraphics():

    def __init__(self, outputPlot):

        self.decayValue = 0

        self.outputPlot = outputPlot

        self.measurementPatch = Rectangle((0,0), 1, 1, linewidth=1, color='gray', facecolor='none')
        self.statePatch = Circle((0,0), 1, facecolor='blue')
        self.speedPatch = Arrow(0,0,3,3)
        self.circlePatch = Circle((0,0),1, linewidth=1,color='red', facecolor='None')
        self.linePatch = ConnectionPatch((0,0),(1,1), 'data')

        self.model = 0
        self.startLine = (0,0)
        self.needToResetStartLine = True

        self.outputPlot.add_patch(self.measurementPatch)
        self.outputPlot.add_patch(self.statePatch)
        self.outputPlot.add_patch(self.speedPatch)
        self.outputPlot.add_patch(self.circlePatch)
        self.outputPlot.add_patch(self.linePatch)

        self.circlePatch.set_visible(False)
        self.linePatch.set_visible(False)

    def setModel(self, model):

        if model != self.model:
            self.decayValue = 10
        elif self.decayValue>0:
            self.decayValue -= 1

            if self.decayValue == 5:
                self.needToResetStartLine = True

        self.model = model

        if self.decayValue <= 0:
            if model==0:
                self.speedPatch.set_visible(False)
                self.linePatch.set_visible(False)
                self.circlePatch.set_visible(False)

            if model==1:
                self.speedPatch.set_visible(True)
                self.linePatch.set_visible(True)
                self.circlePatch.set_visible(False)

            if model==2:
                self.speedPatch.set_visible(True)
                self.linePatch.set_visible(False)
                #self.circlePatch.set_visible(True)

    def setState(self,x,y):
        self.statePatch.center = x, y

    def setMeasurement(self,x,y):
        self.measurementPatch.set_xy((x,y))

    def setSpeed(self, x, y, vx, vy):
        self.speedPatch.remove()
        self.speedPatch = Arrow(x,y, vx/10, vy/10)
        self.outputPlot.add_patch(self.speedPatch)

    def setCircle(self, x, y, R):
        self.circlePatch.set_radius(R)
        self.circlePatch.center = x,y

    def setLine(self, x, y):

        """ if self.needToResetStartLine:
            self.needToResetStartLine = False
            self.startLine = (x,y)

        self.linePatch.remove()
        self.linePatch = ConnectionPatch((self.startLine[0],self.startLine[1]), (x+1, y+1), 'data')
        self.outputPlot.add_patch(self.linePatch) """
        pass

            
                
