import numpy as np
from numpy import matmul
from Constants import *
from LinearModels import MODEL_FUNC

class ParticleIMM:

    def __init__(self):

        self.nmodels = 3
        self.nparticles = 900

        self.states = []
        
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

            self.states.append([])
            
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
                self.states[m].append(s)





    def compute(self, x, y, delta):
        
        #Probabilistic mode change
        self.mix(delta)

        #Prediction
        self.predict(delta)

        #Likelihoods
        self.likelihoodsAndNormalize(x, y)

        means = []

        for m in range(self.nmodels):
            if len(self.states[m]) > 0:
                '''print(self.states[m][0])
                print(self.states[m][1])
                print(np.mean(self.states[m][0:1], axis=0))
                print("______________________")'''
                means.append(np.mean(self.states[m], axis=0))
            else:
                means.append(np.zeros((MODEL_STATESIZE[m], 1)))

            

        #print(means, self.p)
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
            newmodes = np.random.choice(range(self.nmodels), len(self.states[m]), p=pm, replace=True)

            #Browse particles of mode m, make a prediction with a probabilistic mode change
            for i in range(len(self.states[m])):
                
                #Transform the state if mode change
                s = TRANS_FUNC_P[m][newmodes[i]](self.states[m][i])

                newstates[newmodes[i]].append(s)
                #print(s)
         
        self.states = newstates


    def predict(self, delta):

        for m in range(self.nmodels):

            #Optimisation : if model is cvs or void, calculate F and G once
            if m == MODEL_CVS or m == MODEL_VOID:
                F, G = MODEL_FUNC[m](None, delta)

            for i in range(len(self.states[m])):
                
                #Special case of CTR
                if m == MODEL_CTR:
                    F, G = MODEL_FUNC[m](self.states[m][i], delta)

                #Getting process noise
                noise = np.random.randn(MODEL_NOISESIZE[m], 1) * MODEL_SIGMAW[m] * 40

                self.states[m][i] = np.asarray(matmul(F, self.states[m][i]) + matmul(G, noise))


    def likelihoodsAndNormalize(self, x, y):

        z = np.asarray(np.matrix([[x], [y]]))
        
        for m in range(self.nmodels):

            H = MODEL_H[m]
            sigma_n = MODEL_SIGMAN[m]

            logliks = []

            for i in range(len(self.states[m])):
                
                logliks.append(   [np.sum( - np.power(z - matmul(H, self.states[m][i]), 2) /2 /sigma_n )]   )

            if len(logliks) > 0:
                logliks = np.asarray(np.matrix(logliks))

                ml = max(logliks)
                #print(ml)

                self.p[m] = np.sum(np.exp(logliks - ml))
                
                weights = np.exp(logliks - ml)
                weights = (1/np.sum(weights))*weights

                mwidx = np.argmax(weights)
                self.maxstates[m] = self.states[m][mwidx]

                prob = [weights[j][0] for j in range(len(weights))]

                newstateslist = np.random.choice(len(self.states[m]), len(self.states[m]), replace=True, p=prob)
                self.states[m] = [self.states[m][k] for k in newstateslist]

            else:
                self.states[m] = []

        
        self.p = self.p/np.sum(self.p)
    


        
                

