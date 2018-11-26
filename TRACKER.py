from tkinter import *
from TrackerApp import TrackerApp
import threading
from Kalman import KalmanIMM
from Particle import ParticleIMM
from Logger import Logger

class GUI(Frame):

    def __init__(self, filter):
        root = Tk()
        super().__init__(root)

        #Set up exit of all threads when window closed
        exiting = threading.Event()

        #Initialize main window
        #root.geometry('1200x800+32+32')
        self.master.title("Tracker")

        #Setup this frame 
        self.pack(fill=BOTH, expand=1)

        #Initialize components
        self.cam_view = TrackerApp(self, filter)

        #Running Window
        root.mainloop()

        #Terminating program
        self.cam_view.closeCam()
        exiting.set()



#Setting the filter to use, use None if you don't want to log the data
filter = ParticleIMM(Logger())
#filter = KalmanIMM(Logger())
GUI(filter)