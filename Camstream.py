import time
from threading import *
from cv2 import *
from cv2 import CAP_PROP_FRAME_COUNT as CPFC

class Camstream(Thread):

    '''This thread manages the webcam stream and calls TrackerApp.updateImage when an image is available'''

    def __init__(self, camFrame):
        self.isRunning = False
        self.camFrame = camFrame
        self.timer = 0

        #Initializing video stream
        self.videoStream = VideoStream()

        self.startThread()
        


    def startThread(self):
        #Initializing thread
        Thread.__init__(self)

        #Close thread when exiting program
        self.setDaemon(True)

        #Starting Thread
        self.isRunning = True
        self.start()


    def run(self):
        
        self.timer = time.time()

        while(self.isRunning):

            if self.videoStream.availableImage():
                now = time.time()
                delta = now - self.timer
                self.timer = now

                #Send image to TrackerApp
                self.camFrame.updateImage(self.videoStream.getAvailableImage(), delta)
            else:
                time.sleep(0.001)

            

    def closeStream(self):
        self.isRunning = False
        self.videoStream.closeStream()




class VideoStream(Thread):

    '''
    This thread also manages the video stream, buffers the video frame on a separate thread to avoid video latency'''

    def __init__(self):
        self.isRunning = False
        self.image = None
        self.available = False

        #initializing video
        self.cam = VideoCapture(0)
        self.cam.set(3,200)
        self.cam.set(4,200)
        self.cam.set(CAP_PROP_EXPOSURE, 40) 

        self.startThread()

    def getAvailableImage(self):
        if self.available:
            self.available = False
            return self.image
        else:
            return None
        

    def availableImage(self):
        return self.available



    def startThread(self):
        #Initializing thread
        Thread.__init__(self)

        #Close thread when exiting program
        self.setDaemon(True)

        #Starting Thread
        self.isRunning = True
        self.start()

    def run(self):
        while(self.isRunning):
            
            s, img = self.cam.read()

            if s:
                self.image = img
                self.available = True
                waitKey(1)
            else:
                time.sleep(0.0005)

    def closeStream(self):
        self.isRunning = False
        self.cam.release()
        print("cam closed")