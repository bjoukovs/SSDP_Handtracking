import LinearModels
import numpy as np
import Transitions


# Definition of the model names

MODEL_VOID = 0
MODEL_CVS = 1
MODEL_CTR = 2


# Definition of some of the model properties (w = process noise, n = measurement noise)

#Constant velocity straight
CVS_sigma_w = 6
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
VOID_sigma_n = 0

VOID_F = np.ones((2,2))
VOID_QW = np.eye(2)

VOID_H = np.eye(2)
VOID_QN = np.eye(2)*VOID_sigma_n


#Transition matrix
TRANS = np.eye(3)
TRANS_FUNC = [[Transitions.identity, Transitions.void2cvs, Transitions.void2ctr], \
            [Transitions.cvs2void, Transitions.identity, Transitions.cvs2ctr], \
            [Transitions.ctr2void, Transitions.ctr2cvs, Transitions.identity]]
