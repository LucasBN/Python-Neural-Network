import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

Lambda = 0.0001

class Neural_Network(object):
    def __init__(self, Lambda = 0):        
        #Define Hyperparameters
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3
        
        #Weights (parameters)
        self.W1 = np.random.randn(self.inputLayerSize,self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize,self.outputLayerSize)
        
        self.Lambda = Lambda
        
    def forward(self, X):
        #Propogate inputs though network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3) 
        return yHat
        
    def sigmoid(self, z):
        #Apply sigmoid activation function to scalar, vector, or matrix
        return 1/(1+np.exp(-z))
    
    def sigmoidPrime(self,z):
        #Gradient of sigmoid
        return np.exp(-z)/((1+np.exp(-z))**2)
    
    def costFunction(self, X, y):
        #Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)
        J = 0.5*sum((y-self.yHat)**2)/X.shape[0] + (self.Lambda/2)*(np.sum(self.W1**2)+np.sum(self.W2**2))
        return J
        
    def costFunctionPrime(self, X, y):
        #Compute derivative with respect to W and W2 for a given X and y:
        self.yHat = self.forward(X)
        
        delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
        #Add gradient of regularization term:
        dJdW2 = np.dot(self.a2.T, delta3)/X.shape[0] + self.Lambda*self.W2
        
        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
        #Add gradient of regularization term:
        dJdW1 = np.dot(X.T, delta2)/X.shape[0] + self.Lambda*self.W1
        
        return dJdW1, dJdW2
    
    #Helper Functions for interacting with other classes:
    def getParams(self):
        #Get W1 and W2 unrolled into vector:
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params
    
    def setParams(self, params):
        #Set W1 and W2 using single paramater vector.
        W1_start = 0
        W1_end = self.hiddenLayerSize * self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize , self.hiddenLayerSize))
        W2_end = W1_end + self.hiddenLayerSize*self.outputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))
        
    def computeGradients(self, X, y):
        dJdW1, dJdW2 = self.costFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))

def computeNumericalGradient(N, X, y):
        # Gets W1 and W2 as a vector:
        paramsInitial = N.getParams()
        
        # Sets numgrad and pertub equal to the shape of W1 and W2:
        numgrad, perturb = np.zeros(paramsInitial.shape), np.zeros(paramsInitial.shape)
        
        # Epsilon:
        e = 1e-4

        for p in range(len(paramsInitial)):
            
            # perturb[p] becomes 0.0001 (value of e):
            perturb[p] = e 
            
            # Adding a really small number (like we did with epsilon):
            N.setParams(paramsInitial + perturb)
                                        
            # Calculates the cost:
            loss2 = N.costFunction(X, y)
            
            # Subtracting a really small number (like we did with epsilon):
            N.setParams(paramsInitial - perturb)
            
            # Calculates the cost:
            loss1 = N.costFunction(X, y) # calculates the new cost of x and y

            # Compute Numerical Gradient
            numgrad[p] = (loss2 - loss1) / (2*e)

            # Return the value we changed to zero:
            perturb[p] = 0
            
        # Return Params to original value:
        N.setParams(paramsInitial)

        return numgrad

class trainer(object):
    def __init__(self, N):
        self.N = N
    def costFunctionWrapper(self, params, x, y):
        self.N.setParams(params)
        cost = self.N.costFunction(x, y)
        grad = self.N.computeGradients(x, y)
        return cost, grad
    def callbackF(self, params):
        self.N.setParams(params)
        self.J.append(self.N.costFunction(self.x, self.y))
    def train(self, x, y):
        self.x = x
        self.y = y
        
        self.J = []
        params0 = self.N.getParams()
        options = {'maxiter': 200, 'disp': True}
        
        _res = optimize.minimize(self.costFunctionWrapper, params0, jac = True, method = 'BFGS', args = (x, y), options = options, callback = self.callbackF)
        self.N.setParams(_res.x)
        self.optimizationResults = _res

NN = Neural_Network(Lambda = 0.0001)
T = trainer(NN)

