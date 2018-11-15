import numpy as np
from math import cos, sin
from Constants import *


#Constant velocity straight line model
def cvs(x, delta):

    # Returns the linearized state update matrix
    # x = [x y vx vy]
    # delta = Time delay

    # State equation
    F = np.matrix([[1, 0, delta, 0], \
        [0, 1, 0, delta], \
        [0, 0, 1, 0], \
        [0, 0, 0, 1]])

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

    """ theta_est = x[2][0]
    omega_est = x[3][0]
    R_est = x[4][0]
    
    F = np.matrix([[1, 0, 0, -R_est*sin(theta_est)*delta, 0], \
            [0, 1, 0, R_est*cos(theta_est)*delta, 0], \
            [0, 0, 1, delta, 0], \
            [0, 0, 0, 1, 0], \
            [0, 0, 0, 0, 1]])


    G = np.matrix([[-delta**2/2*sin(theta_est)*R_est, delta*cos(theta_est)*R_est],  \
            [delta**2/2*cos(theta_est)*R_est, delta*sin(theta_est)*R_est],  \
            [delta**2/2, 0],  \
            [delta, 0],  \
            [0, delta]])

    return F,G """

    # x = [x y vx vy omega]
    vx_est = x[2][0]
    vy_est = x[3][0]
    omega_est = x[4][0]

    if omega_est==0: omega_est = 0.0001

    si = sin(omega_est*delta)
    co = cos(omega_est*delta)


    F = np.matrix( [[1, 0, si/omega_est, (co-1)/omega_est, 0], \
                [0, 1, (1-co)/omega_est, si/omega_est, 0], \
                [0, 0, co, -si, 0], \
                [0, 0, si, co, 0],\
                [0, 0, 0, 0, 1]]  )
        
    G = np.matrix( [[delta**2/2, 0, delta**2/2] ,\
                [0, delta**2/2, delta**2/2],
                [delta, 0, delta],
                [0, delta, delta],
                [0, 0, delta]])
    
    return F, G





#Object absent : Only noise
def no_object(x, delta):
    F = VOID_F
    G = np.eye(2)*delta**2/2

    return F,G
