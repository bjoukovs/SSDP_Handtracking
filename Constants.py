import LinearModels
import numpy as np
import Transitions


# Definition of the model names

MODEL_VOID = 0
MODEL_CVS = 1
MODEL_CTR = 2


# Definition of some of the model properties (w = process noise, n = measurement noise)

#Constant velocity straight
CVS_sigma_w = 5
CVS_sigma_n = 1
CVS_QW = np.eye(2)*CVS_sigma_w

CVS_H = np.zeros((2,4))
CVS_H[0][0] = 1
CVS_H[1][1] = 1

CVS_QN = np.eye(2)*CVS_sigma_n

#Constant turn rate
CTR_sigma_w = 6
CTR_sigma_n = 1
CTR_QW = np.eye(2)*CTR_sigma_w

CTR_H = np.zeros((2,5))
CTR_H[0][0] = 1
CTR_H[1][1] = 1

CTR_QN = np.eye(2)*CTR_sigma_n

#Only noise
VOID_sigma_n = 10

VOID_F = np.eye(2)
VOID_QW = np.eye(2)

VOID_H = np.eye(2)
VOID_QN = np.eye(2)*VOID_sigma_n


#Transition matrix
TRANS_FUNC = [[Transitions.identity, Transitions.void2cvs, Transitions.void2ctr], \
            [Transitions.cvs2void, Transitions.identity, Transitions.cvs2ctr], \
            [Transitions.ctr2void, Transitions.ctr2cvs, Transitions.identity]]


P_VOID2VOID = 0.9
P_VOID2CVS = 0.05
P_VOID2CTR = 0.05

P_CTR2CTR = 0.9
P_CTR2CVS = 0.05
P_CTR2VOID = 0.05

P_CVS2CVS = 0.9
P_CVS2CTR = 0.05
P_CVS2VOID = 0.05

TRANS = [[P_VOID2VOID, P_VOID2CVS, P_VOID2CTR], \
        [P_CVS2VOID, P_CVS2CVS, P_CVS2CTR], \
        [P_CTR2VOID, P_CTR2CVS, P_CTR2CTR]]