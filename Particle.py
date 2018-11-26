import numpy as np
from numpy import matmul
from numpy import matlib
from Constants import *
from LinearModels import MODEL_FUNC
from math import sqrt, pi

class ParticleIMM:

    def __init__(self, logger):

        self.nmodels = 3
        self.nparticles = 1800

        self.states = []

        self.logger = logger
        
        self.resetState(100,100)




    def resetState(self, x, y):
        
        #States initialization
        self.states = []
        self.maxstates = [[], [], []]
        self.p = np.ones((self.nmodels, 1))/self.nmodels

        ppstate = round(self.nparticles/self.nmodels)

        initialState = np.zeros((2,1))
        initialState[0] = x
        initialState[0] = y

        for m in range(self.nmodels):

            #self.states.append([])
            self.states.append(np.zeros((MODEL_STATESIZE[m], ppstate)))
            
            for i in range(ppstate):
                #Get particle for model from the initial state
                s = TRANS_FUNC_P[0][m](initialState)  

                #Adding randomness
                if m==MODEL_VOID:
                    noise = np.multiply(np.random.randn(2,1), VOID_PARTCLE_UNCERTAINTY)
                elif m==MODEL_CVS:
                    noise = np.multiply(np.random.randn(4,1), CVS_PARTCLE_UNCERTAINTY)
                elif m==MODEL_CTR:
                    noise = np.multiply(np.random.randn(5,1), CTR_PARTCLE_UNCERTAINTY)

                s = s + noise
                #self.states[m].append(s)
                self.states[m][:,i] = s[:,0]
                





    def compute(self, x, y, delta):
        
        #Probabilistic mode change
        self.mix(delta)

        #Prediction
        self.predict(delta)

        #Likelihoods
        self.likelihoodsAndNormalize(x, y)

        means = []

        for m in range(self.nmodels):
            if self.states[m].shape[1] > 0:
                '''print(self.states[m][0])
                print(self.states[m][1])
                print(np.mean(self.states[m][0:1], axis=0))
                print("______________________")'''
                means.append(np.mean(self.states[m], axis=1))
            else:
                means.append(np.zeros((MODEL_STATESIZE[m], 1)))

            
        if self.logger is not None:
            for m in range(self.nmodels):
                self.logger.write('means'+str(m), means[m])
                self.logger.write('std'+str(m), np.std(self.states[m], axis=1))

            self.logger.write('p', self.p)
            self.logger.write('meas', [x, y, delta])

        return means, self.p
        #return self.maxstates, self.p



    def mix(self, delta):

        #Initialize new states array
        newstates = []
        for m in range(self.nmodels):
            newstates.append([])

        for m in range(self.nmodels):

            #Mode change probabilities
            pm = [TRANS[m][j] for j in range(self.nmodels)]

            #New modes for these particles
            #newmodes = np.random.choice(range(self.nmodels), len(self.states[m]), p=pm, replace=True)
            newmodes = np.random.choice(range(self.nmodels), self.states[m].shape[1], p=pm, replace=True)

            #Browse particles of mode m, make a prediction with a probabilistic mode change
            #for i in range(len(self.states[m])):
            for i in range(self.states[m].shape[1]):
                
                #Transform the state if mode change
                #s = TRANS_FUNC_P[m][newmodes[i]](self.states[m][i])
                s = TRANS_FUNC_P[m][newmodes[i]](np.expand_dims(self.states[m][:,i], axis=1))

                newstates[newmodes[i]].append(s)
                #print(s)
        
        #self.states = newstates
        for m in range(self.nmodels):
            self.states[m] = np.concatenate(newstates[m], axis=1)
        
        


    def predict(self, delta):

        for m in range(self.nmodels):

            #Optimisation : if model is cvs or void, calculate F and G once and perform broadcasting
            if m == MODEL_CVS or m == MODEL_VOID:
                F, G = MODEL_FUNC[m](None, delta)
                noise = np.random.randn(MODEL_NOISESIZE[m], self.states[m].shape[1]) * sqrt(MODEL_varW[m])*5
                self.states[m] = np.asarray(matmul(F, self.states[m]) + matmul(G, noise))

            #Special case of CTR
            else:

                for i in range(self.states[m].shape[1]):

                    s = np.expand_dims(self.states[m][:,i], axis=1)
                    
                    F, G = MODEL_FUNC[m](s, delta)

                    #Getting process noise
                    noise = np.random.randn(MODEL_NOISESIZE[m], 1) * sqrt(MODEL_varW[m])*5

                    self.states[m][:,i] = np.asarray(matmul(F, s) + matmul(G, noise))[:,0]

                    


    def likelihoodsAndNormalize(self, x, y):

        z = np.asarray(np.matrix([[x], [y]]))

        prob = []
        
        for m in range(self.nmodels):

            prob.append([])

            H = MODEL_H[m]
            var_n = MODEL_varN[m]

            if self.states[m].shape[1]>0:
                z_tilde = -matmul(MODEL_H[m], self.states[m]) + matlib.repmat(z, 1, self.states[m].shape[1])
                Qn_inv = np.linalg.inv(MODEL_QN[m])
                detQ = np.linalg.det(MODEL_QN[m])
                logliks = np.log(1/sqrt(pi**2*detQ))-0.5*np.sum(z_tilde * matmul(Qn_inv, z_tilde), axis=0)
                #logliks = np.sum( - np.power(z - matmul(H, self.states[m]), 2) /2 /var_n, axis=0)

            #for i in range(self.states[m].shape[1]):
                
                #logliks.append(   [np.sum( - np.power(z - matmul(H, self.states[m][i]), 2) /2 /var_n )]   )

            #if len(logliks) > 0:
                #logliks = np.asarray(np.matrix(logliks))

                ml = max(logliks)
                #print(ml)

                self.p[m] = np.sum(np.exp(logliks - ml))
                
                weights = np.exp(logliks - ml)
                weights = (1/np.sum(weights))*weights

                mwidx = np.argmax(weights)
                self.maxstates[m] = np.expand_dims(self.states[m][:,mwidx], axis=1)

                #prob[m] = [weights[j][0] for j in range(len(weights))]
                prob[m] = [weights[j] for j in range(len(weights))]

            else:
                prob[m] = 0



        self.p = self.p/np.sum(self.p)

        #Resampling
        for m in range(self.nmodels):

            nstatesinmodel = int(round(self.nparticles*self.p[m][0]))

            #resamplestates = np.zeros((MODEL_STATESIZE[m], ntatesinmodel))
            #print(prob[m])
            newstateslist = np.random.choice(self.states[m].shape[1], nstatesinmodel, replace=True, p=prob[m])
            #self.states[m] = [self.states[m][k] for k in newstateslist]
            #print(self.states[m][:,newstateslist])
            self.states[m] = self.states[m][:,newstateslist]

        #print(len(self.states[0]) + len(self.states[1]) + len(self.states[2]))
        
        
    


        
                

