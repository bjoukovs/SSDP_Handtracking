from tkinter import *
from Camview import Camview
import threading

class GUI(Frame):

    def __init__(self):
        root = Tk()
        super().__init__(root)

        #Set up exit of all threads when window closed
        exiting = threading.Event()

        #Initialize main window
        root.geometry('800x600')
        self.master.title("Hand Tracker")

        #Setup this frame 
        self.pack(fill=BOTH, expand=1)

        #Initialize components
        self.cam_view = Camview(self)

        #Running Window
        root.mainloop()

        #Terminating program
        self.cam_view.closeCam()
        exiting.set()


