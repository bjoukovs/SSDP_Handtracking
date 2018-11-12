from tkinter import *
from TrackerApp import TrackerApp
import threading
from Kalman import IMM

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



#Setting the filter to use
filter = IMM()
GUI(filter)