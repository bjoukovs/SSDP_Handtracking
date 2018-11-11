import numpy as np
from math import cos, sin, asin, acos, sqrt

def cvs2ctr(s, Q):

    #Transition of constant velocity state to constant turn state
    s2 = np.ndarray((5,1))
    s2[0:1] = s[0:1]
    s2[2] = 0
    s2[3] = 1
    s2[4] = 10  #TO VERIFY, replace by priors

    Q2 = np.eye(5)
    Q2[0][0] = Q[0][0]
    Q2[1][1] = Q[1][1]

    return s2, Q2




def ctr2cvs(s, Q):

    s2 = np.ndarray((4,1))
    s2[0:1] = s[0:1]
    s2[2] = -s[4]*s[3]*sin(s[2])
    s2[3] = s[4]*s[3]*cos(s[2])

    Q2 = np.eye(4)
    Q2[0][0] = Q[0][0]
    Q2[1][1] = Q[1][1]
    #Q2[2][2] =     TO VERIFY

    return s2, Q2

def cvs2void(s, Q):
    s2 = np.ndarray((2,1))
    s2[0:1] = s[0:1]

    Q2 = np.eye(2)

    return s2, Q2

def void2cvs(s,Q):
    s2 = np.ndarray((4,1))
    s2[0:1] = s[0:1]
    s2[2] = 0
    s2[3] = 0

    Q2 = np.eye(4)
    Q2[0][0] = Q[0][0]
    Q2[1][1] = Q[1][1]
    #Q2[2][2] =     TO VERIFY

    return s2, Q2

def ctr2void(s,Q):
    return cvs2void(s, Q)


def void2ctr(s,Q):

    s2 = np.ndarray((5,1))
    s2[0:1] = s[0:1]
    s2[2] = 0
    s2[3] = 1
    s2[4] = 10  #TO VERIFY, replace by priors

    Q2 = np.eye(5)
    Q2[0][0] = Q[0][0]
    Q2[1][1] = Q[1][1]

    return s2, Q2



def identity(s, Q):
    return s, Q