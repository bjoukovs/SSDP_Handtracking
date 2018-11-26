import numpy as np
import datetime
import os

class Logger():

    def __init__(self):
        
        now = str(datetime.datetime.now()).replace(':','_').replace('.','_')
        self.basefilename = now

        logdir = os.getcwd()+"/log/foo.txt"
        print(os.path.realpath(logdir))
        try:
            os.mkdir(os.path.dirname(logdir))
        except:
            pass

    def write(self, name, data):
        name = os.getcwd()+"/log/log_"+name+"_"+self.basefilename+".csv"

        file = open(name, 'ab')
        np.savetxt(file, data, delimiter=",", footer=";")
        file.close()