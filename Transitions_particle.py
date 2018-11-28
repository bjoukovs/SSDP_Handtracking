import numpy as np
from math import cos, sin, asin, acos, sqrt

VMAX = 400
OMEGAMAX = 16

'''
Transition functions for the Particle filter. Avoids computing Q as in the Kalman case, and can use priors on state variables
'''

def cvs2ctr(s):
    s = np.asarray(s)
    
    s2 = np.ndarray((5,1))
    s2[0] = s[0]
    s2[1] = s[1]
    s2[2] = s[2]
    s2[3] = s[3]
    s2[4] = (-1)**np.random.randint(0,2)*10 + np.random.randn(1,1)*5

    return s2



def ctr2cvs(s):

    s = np.asarray(s)

    s2 = np.ndarray((4,1))
    s2[0] = s[0]
    s2[1] = s[1]
    s2[2] = s[2]
    s2[3] = s[3]

    return s2


def cvs2void(s):
    s = np.asarray(s)

    s2 = np.ndarray((2,1))
    s2[0] = s[0]
    s2[1] = s[1]

    return s2

def void2cvs(s):
    s = np.asarray(s)

    s2 = np.ndarray((4,1))
    s2[0] = s[0]
    s2[1] = s[1]
    s2[2] = np.random.randn(1,1)*10
    s2[3] = np.random.randn(1,1)*10

    return s2

def ctr2void(s):
    return cvs2void(s)


def void2ctr(s):
    s = np.asarray(s)
    
    s2 = np.ndarray((5,1))
    s2[0] = s[0]
    s2[1] = s[1]
    s2[2] = np.random.randn(1,1)*10
    s2[3] = np.random.randn(1,1)*10
    s2[4] = (-1)**np.random.randint(0,2)*10 + np.random.randn(1,1)*5

    return s2



def identity(s):
    return s