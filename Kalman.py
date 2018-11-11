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
        F,G = LinearModels.ctr(s, delta)
        QW = CTR_QW

    if model == MODEL_VOID:
        F,G = VOID_F,np.eye(2)
        QW = VOID_QW
        

    s2 = matmul(F,s)
    Q2 = matmul(F, matmul(Q, F.transpose())) + matmul(G, matmul(QW, G.transpose()))
    return s2, Q2

    
def update(s, z, Q, model, delta):
    #Returns su, Qu

    if model == MODEL_CVS:
        H = CVS_H
        QN = CVS_QN

    if model == MODEL_CTR:
        H = CTR_H
        QN = CTR_QN

    if model == MODEL_VOID:
        H = VOID_H
        QN = VOID_QN

    #Kalman gain
    K = matmul(Q, matmul(   H.transpose(), np.linalg.inv(   matmul(H, matmul(Q, H.transpose()))  +  QN   )   ))

    s2 = s + matmul(K, z - matmul(H, s))
    Q2 = Q - matmul(K, matmul(H, Q))

    return s2, Q2




class IMM:

    def __init__(self):

        self.s = np.zeros((5,1))
        self.Q = np.eye(5)

        self.nModels = 3

        self.states = []
        self.states.append(np.zeros((2,1))) # 0 = VOID
        self.states.append(np.zeros((4,1))) # 1 = CVS
        self.states.append(np.zeros((5,1))) # 2 = CTR
        

        # !! SEE MATLAB FOR CORRECT INITIALIZATION
        self.covariances = []
        self.covariances.append(np.eye(2))
        self.covariances.append(np.eye(4))
        self.covariances.append(np.eye(5))

        # Likelihoods
        self.likelihoods = np.zeros((3,1))

        self.p = np.ones((self.nModels, 1))/self.nModels



    def compute(self, x, y, delta):
        #Function called when new measurement received
        z = np.matrix([[x], [y]])

        #MIXING
        sfm, Qfm, pm = self.mix()

        #PREDICTION
        for i in range(self.nModels):
            s, Q = predict(sfm[i], Qfm[i], i, delta)
            self.states[i] = s
            self.covariances[i] = Q

        #UPDATE
        for i in range(self.nModels):
            s, Q, l = update(self.states[i], z, self.covariances[i], i, delta)
            self.states[i] = np.asarray(s)
            self.covariances[i] = np.asarray(Q)
            self.likelihoods[i] = l

        #UPDATE PROBABILITIES (see imm_upd.m)
        for i in range(self.nModels):
            pass

        #prediction
        #s, Q = predict(self.s, self.Q, MODEL_CTR, delta)
        #s, Q = update(s, z, Q, MODEL_CTR, delta)

        #self.s = s
        #self.Q = Q

        print(pm)

        return s

    def mix(self):

        pm = matmul(TRANS, self.p)
        sfm = []
        Qfm = []

        for i in range(self.nModels):
            sfm.append(np.zeros(self.states[i].shape))
            Qfm.append(np.zeros(self.covariances[i].shape))
            
            for j in range(self.nModels):
                #Call transition function
                s, Q = TRANS_FUNC[j][i](self.states[j], self.covariances[j])
                
                #Mixing
                sfm[i] = sfm[i] + s * TRANS[i][j] * self.p[j]
                Qfm[i] += np.asarray(Q + s * s.transpose()) * TRANS[i][j] * self.p[j]

            sfm[i] /= pm[i]
            Qfm[i] = Qfm[i]/pm[i] - matmul(sfm[i], sfm[i].transpose())

        return sfm, Qfm, pm
 

