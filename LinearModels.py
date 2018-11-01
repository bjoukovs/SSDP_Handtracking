import numpy as np
from math import cos, sin
from Constants import *


#Constant velocity straight line model
def cvs(x, delta):

    # Returns the linearized state update matrix
    # x = [x y vx vy]
    # delta = Time delay

    # State equation
    F = np.eye(4,4)
    F[0][2] = delta
    F[1][3] = delta

    #Process noise matrix G
    G = np.matrix([[delta**2/2, 0],  \
            [0, delta**2/2],  \
            [delta, 0],  \
            [0, delta]])

    return F,G



#Constant turn rate circle
def ctr(x, delta):

    # Returns the linearized state update matrix
    # x = [x y theta omega R]
    # delta = Time delay

    theta_est = x[2]
    omega_est = x[3]
    R_est = x[4]
    
    F = np.matrix([[1, 0, 0, -R_est*sin(theta_est)*delta, 0], \
            [0, 1, 0, R_est*cos(theta_est)*delta, 0], \
            [0, 0, 1, delta, 0], \
            [0, 0, 0, 1, 0], \
            [0, 0, 0, 0, 1]])

    G = np.matrix([[delta**2/2*sin(theta_est), delta*cos(theta_est)],  \
            [delta**2/2*cos(theta_est), delta*sin(theta_est)],  \
            [delta**2/2, 0],  \
            [delta, 0],  \
            [0, delta]])

    return F,G


#Object absent : Only noise
def no_object():
    pass
