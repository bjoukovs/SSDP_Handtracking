import LinearModels
import numpy as np
import Transitions
import LinearModels


# Definition of some of the model properties (w = process noise, n = measurement noise)

#Constant velocity straight
CVS_sigma_w = 0.02
CVS_sigma_n = 0.5
CVS_QW = np.eye(2)*CVS_sigma_w

CVS_H = np.zeros((2,4))
CVS_H[0][0] = 1
CVS_H[1][1] = 1

CVS_QN = np.eye(2)*CVS_sigma_n

CVS_PARTCLE_UNCERTAINTY = np.zeros((4,1))
CVS_PARTCLE_UNCERTAINTY[0][0] = 100
CVS_PARTCLE_UNCERTAINTY[1][0] = 100
CVS_PARTCLE_UNCERTAINTY[2][0] = 10
CVS_PARTCLE_UNCERTAINTY[3][0] = 10

#Constant turn rate
CTR_sigma_w = 0.3
CTR_sigma_n = 1
CTR_QW = np.eye(3)*CTR_sigma_w

CTR_H = np.zeros((2,5))
CTR_H[0][0] = 1
CTR_H[1][1] = 1

CTR_QN = np.eye(2)*CTR_sigma_n

CTR_PARTCLE_UNCERTAINTY = np.zeros((5,1))
CTR_PARTCLE_UNCERTAINTY[0][0] = 100
CTR_PARTCLE_UNCERTAINTY[1][0] = 100
CTR_PARTCLE_UNCERTAINTY[2][0] = 10
CTR_PARTCLE_UNCERTAINTY[3][0] = 10
CTR_PARTCLE_UNCERTAINTY[4][0] = 10

#Noise - Stationnary
VOID_sigma_w = 0.001
VOID_sigma_n = 1


VOID_F = np.eye(2)
VOID_QW = np.eye(2)*VOID_sigma_w

VOID_H = np.eye(2)
VOID_QN = np.eye(2)*VOID_sigma_n

VOID_PARTCLE_UNCERTAINTY = np.zeros((2,1))
VOID_PARTCLE_UNCERTAINTY[0][0] = 100
VOID_PARTCLE_UNCERTAINTY[1][0] = 100


#Transition matrix
TRANS_FUNC = [[Transitions.identity, Transitions.void2cvs, Transitions.void2ctr], \
            [Transitions.cvs2void, Transitions.identity, Transitions.cvs2ctr], \
            [Transitions.ctr2void, Transitions.ctr2cvs, Transitions.identity]]


P_VOID2VOID = 0.9
P_VOID2CVS = 0.05
P_VOID2CTR = 0.05

P_CTR2CTR = 0.96
P_CTR2CVS = 0.039
P_CTR2VOID = 0.001

P_CVS2CVS = 0.98
P_CVS2CTR = 0.019
P_CVS2VOID = 0.001

TRANS = np.asarray(np.matrix([[P_VOID2VOID, P_VOID2CVS, P_VOID2CTR], \
        [P_CVS2VOID, P_CVS2CVS, P_CVS2CTR], \
        [P_CTR2VOID, P_CTR2CVS, P_CTR2CTR]]))

# Definition of the model 
# names

MODEL_VOID = 0
MODEL_CVS = 1
MODEL_CTR = 2

MODEL_H = [VOID_H, CVS_H, CTR_H]
MODEL_SIGMAN = [VOID_sigma_n, CVS_sigma_n, CTR_sigma_n]
MODEL_SIGMAW = [VOID_sigma_w, CVS_sigma_w, CTR_sigma_w]
MODEL_STATESIZE = [2, 4, 5]
MODEL_NOISESIZE = [2, 2, 3]