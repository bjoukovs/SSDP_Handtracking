from tkinter import *
from Camstream import Camstream
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import cv2
from Handmatcher import Handmatcher
import numpy as np
import time

class Camview(Frame):

    def __init__(self, mainFrame):
        super().__init__(mainFrame)

        self.pack(side = LEFT)

        #initialize matplotlib figure for image
        self.imFigure = Figure(figsize=(10,5), dpi=100)
        self.imPlot = self.imFigure.add_subplot(121)
        self.imPlot2 = self.imFigure.add_subplot(122)
        self.imCanvas = FigureCanvasTkAgg(self.imFigure, self)
        self.imCanvas.get_tk_widget().pack(side=BOTTOM, fill=BOTH, expand=True)

        #detection settings
        self.frames = 0
        self.dtreshold = 0

        #Initialize handmatcher
        self.handmatcher = Handmatcher()

        #initialize camera
        self.running = True

        self.camstream = Camstream(self)

        


    def updateImage(self, img):

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

        #plotting image
        self.imPlot.clear()
        self.imPlot.imshow(img_backup, interpolation=None)


        #plotting matching map
        self.imPlot2.clear()
        correlated = self.handmatcher.correlate(img)

        in_val, max_val, min_loc, max_loc = cv2.minMaxLoc(correlated)
        rect2 = Rectangle(max_loc,5,5,linewidth=1,edgecolor='r',facecolor='none')

        if self.frames<10:
            self.frames+=1
            self.dtreshold += max_val/10
        else:
            res, thr = cv2.threshold(correlated,self.dtreshold*0.9,max_val,cv2.THRESH_TOZERO)
            self.imPlot2.imshow(thr, interpolation=None)
            self.imPlot2.add_patch(rect2)


        self.imPlot2.axis("off")
        self.imPlot.axis("off")
        self.imCanvas.draw()

    def closeCam(self):
        self.camstream.closeStream()