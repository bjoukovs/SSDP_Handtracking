import LinearModels
from Constants import *
from numpy import matmul
import numpy as np


def predict(s, Q, model, delta):
    #Returns sp, qp

    if model == MODEL_CVS:
        F,G = LinearModels.cvs(s, delta)
        QW = CVS_QW

    if model == MODEL_CTR:
        F,G = LinearModels.cvs(s, delta)
        QW = CTR_QW
        

    s = matmul(F,s)
    Q = matmul(F, matmul(Q, F.transpose())) + matmul(G, matmul(QW, G.transpose()))

    return s, Q

    
def update(s, z, Q, model, delta):
    #Returns su, Qu

    if model == MODEL_CVS:
        H = CVS_H
        QN = CVS_QN

    if model == MODEL_CTR:
        H = CTR_H
        QN = CTR_QN

    #Kalman gain
    print(matmul(QN, H.transpose()))
    K = matmul(Q, matmul(   H.transpose(), np.linalg.inv(   matmul(H, matmul(QN, H.transpose()))  +  QN   )   ))

    s = s + matmul(K, z - matmul(H, s))
    Q = Q - matmul(K, matmul(H, Q))

    return s, Q




class IMM:

    def __init__(self):

        self.s = np.zeros((4,1))
        self.Q = np.eye(4)

    def compute(self, x, y, delta):
        #Function called when new measurement received
        z = np.matrix([[x], [y]])

        #prediction
        s, Q = predict(self.s, self.Q, MODEL_CVS, delta)
        s, Q = update(s, z, Q, MODEL_CVS, delta)

        self.s = s
        self.Q = Q

        return s[0][0], s[1][0]
