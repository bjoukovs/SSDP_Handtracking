import numpy as np
from math import cos, sin, asin, acos, sqrt

VMAX = 400
OMEGAMAX = 16


'''
This file contains the transition functions, computing s and Q from one model to the other
Use with the IMM Kalman filter
'''

def cvs2ctr(s, Q):
    s = np.asarray(s)
    Q = np.asarray(Q)

    #Transition of constant velocity state to constant turn state

    s2 = np.ndarray((5,1))
    s2[0] = s[0]
    s2[1] = s[1]
    s2[2] = s[2]
    s2[3] = s[3]
    #s2[4] = np.random.randn(1,1)*10
    s2[4] = -6

    Q2 = np.eye(5)
    Q2[0][0] = Q[0][0]
    Q2[1][1] = Q[1][1]
    Q2[2][2] = Q[2][2]
    Q2[3][3] = Q[3][3]
    Q2[4][4] = OMEGAMAX**2/3

    return s2, Q2



def ctr2cvs(s, Q):

    #Transition from CTR to CVS

    s = np.asarray(s)
    Q = np.asarray(Q)

    s2 = np.ndarray((4,1))
    s2[0] = s[0]
    s2[1] = s[1]
    s2[2] = s[2]
    s2[3] = s[3]

    Q2 = np.eye(4)
    Q2[0][0] = Q[0][0]
    Q2[1][1] = Q[1][1]
    Q2[2][2] = Q[2][2]
    Q2[3][3] = Q[3][3]

    return s2, Q2


def cvs2void(s, Q):

    #CVS to VOID

    s = np.asarray(s)
    Q = np.asarray(Q)

    s2 = np.ndarray((2,1))
    s2[0] = s[0]
    s2[1] = s[1]

    Q2 = np.zeros((2,2))
    Q2[0][0] = Q[0][0]
    Q2[1][1] = Q[1][1]

    return s2, Q2

def void2cvs(s,Q):

    #Void tu CVS

    s = np.asarray(s)
    Q = np.asarray(Q)

    s2 = np.ndarray((4,1))
    s2[0] = s[0]
    s2[1] = s[1]
    s2[2] = 0
    s2[3] = 0


    Q2 = np.eye(4)
    Q2[0][0] = Q[0][0]
    Q2[1][1] = Q[1][1]
    Q2[2][2] = VMAX**2/3
    Q2[3][3] = VMAX**2/3

    return s2, Q2

def ctr2void(s,Q):

    #CTR to VOID

    return cvs2void(s, Q)


def void2ctr(s,Q):

    #VOIR to CTR

    s = np.asarray(s)
    Q = np.asarray(Q)

    s2 = np.ndarray((5,1))
    s2[0] = s[0]
    s2[1] = s[1]
    s2[2] = 0
    s2[3] = 0
    s2[4] = -6

    Q2 = np.eye(5)
    Q2[0][0] = Q[0][0]
    Q2[1][1] = Q[1][1]
    Q2[2][2] = VMAX**2/3
    Q2[3][3] = VMAX**2/3
    Q2[4][4] = OMEGAMAX**2/3

    return s2, Q2



def identity(s, Q):
    return s, Q