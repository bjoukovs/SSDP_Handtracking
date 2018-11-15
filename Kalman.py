import LinearModels
from Constants import *
from numpy import matmul
import numpy as np
from math import pi, sqrt
from numpy import exp


def predict(s, Q, model, delta):
    #Returns sp, qp

    if model == MODEL_CVS:
        F,G = LinearModels.cvs(s, delta)
        QW = CVS_QW

    if model == MODEL_CTR:
        F,G = LinearModels.ctr(s, delta)
        QW = CTR_QW

    if model == MODEL_VOID:
        F,G = LinearModels.no_object(s, delta)
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
    V = matmul(H, matmul(Q, H.transpose())) + QN
    Vinv = np.linalg.inv(V)

    K = matmul(Q, matmul(H.transpose(), Vinv))


    # s and Q update
    zp = matmul(H,s)
    z_tilde = z-zp
    s2 = s + matmul(K, z_tilde)
    Q2 = Q - matmul(K, matmul(H, Q))

    #Computing model log-likelihood
    n = max(z.shape)
    l  = np.log( 1/sqrt((2*pi)**n * np.linalg.det(V)) ) +  matmul(   -0.5 * z_tilde.transpose(), matmul(Vinv, z_tilde) )

    return s2, Q2, l





class IMM:

    def __init__(self):

        self.nModels = 3
        self.states = []
        self.covariances = []

        self.resetState(0,0)
        



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
            
        plog = self.likelihoods + np.log(pm)
        plog = plog - np.max(plog)*np.ones(plog.shape)

        p = np.exp(plog)
        self.p = (1/np.sum(p))*p #normalizing

        #prediction
        #s, Q = predict(self.s, self.Q, MODEL_CTR, delta)
        #s, Q = update(s, z, Q, MODEL_CTR, delta)

        #self.s = s
        #self.Q = Q

        return self.states, self.p



    def mix(self):

        pm = matmul(TRANS, self.p)
        sfm = []
        Qfm = []

        for i in range(self.nModels):

            if pm[i] > 0.000001:
                sfm.append(np.zeros(self.states[i].shape))
                Qfm.append(np.zeros(self.covariances[i].shape))
                
                for j in range(self.nModels):
                    #Call transition function
                    s, Q = TRANS_FUNC[j][i](self.states[j], self.covariances[j])
                    
                    #Mixing
                    sfm[i] = sfm[i] + s * TRANS[i][j] * self.p[j]
                    Qfm[i] = Qfm[i] + np.asarray(Q + s * s.transpose()) * TRANS[i][j] * self.p[j]

                sfm[i] = sfm[i]/pm[i]
                Qfm[i] = Qfm[i]/pm[i] - matmul(sfm[i], sfm[i].transpose())

            else:
                sfm[i] = self.states[i]
                Qfm[i] = self.covariances[i]

        return sfm, Qfm, pm


    def resetState(self, x, y):
        
        #States and covariances initialization
        self.states = []
        self.states.append(np.asarray(np.matrix([[x],[y]]))) # 0 = VOID

        self.covariances = []
        self.covariances.append(np.eye(2)) #0 = VOID

        for i in range(self.nModels-1):
            s, Q = TRANS_FUNC[0][i+1](self.states[0], self.covariances[0])
            self.states.append(s)
            self.covariances.append(Q)

        # Log-Likelihoods
        self.likelihoods = np.zeros((3,1))
        self.p = np.ones((self.nModels, 1))/self.nModels
 

