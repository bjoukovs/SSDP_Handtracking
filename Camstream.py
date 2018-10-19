import time
from threading import *
from cv2 import *
from cv2 import CAP_PROP_FRAME_COUNT as CPFC

class Camstream(Thread):

    def __init__(self, camFrame):
        self.isRunning = False
        self.camFrame = camFrame

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
        while(self.isRunning):

            if self.videoStream.availableImage():
                
                a=time.time()
                self.camFrame.updateImage(self.videoStream.getAvailableImage())
                #time.sleep(0.05)
                print(time.time()-a)

            

    def closeStream(self):
        self.isRunning = False
        self.videoStream.closeStream()




class VideoStream(Thread):

    def __init__(self):
        self.isRunning = False
        self.image = None

        #initializing video, 100x100
        self.cam = VideoCapture(0)
        self.cam.set(3,200)
        self.cam.set(4,200)
        self.cam.set(CAP_PROP_EXPOSURE, 40) 

        self.startThread()

    def getAvailableImage(self):
        return self.image

    def availableImage(self):
        if self.image is not None:
            return True
        else:
            return False



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
                waitKey(1)

    def closeStream(self):
        self.isRunning = False
        self.cam.release()
        print("cam closed")