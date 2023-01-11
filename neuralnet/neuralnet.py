import sys
import numpy as np
from sklearn.preprocessing import OneHotEncoder

class MLP(object):
    def __init__(self, no_hidden=100, eta=0.01, random_state=12):
        self.d = no_hiddens
        self.eta = eta
        self.rs = np.RandomState(random_state)
        self.oh_encoder = OneHotEncoder(sparse=False)
        
    def forward(self, X):
        '''
        forward propagation
        '''
        self.A_in = self._create_A_in(X)
        self.Z_h = self.A_in.dot(self.W_h)
        self.A_h = self.sigmoid(self.Z_h)
        self.Z_out = self.A_h.dot(self.W_out)
        self.A_out = self.softmax(self.Z_out)
        return self.A_out
        
        
    def fit(self, X, y):
        '''
        fitting data with X and y
        '''
        y = self.onehot(y)
        self.t = y.shape[1]
        self.A_in = self._init_A_in(X)
        self.W_h = self._init_W_h((self.m, self.d))

    def loss(self, y, ypredict_proba):
        return - np.sum(y * np.log(ypredict_proba))
        
    def _init_A_in(self, X):
        bias = np.ones((X.shape[0], 1))
        final = np.c_[bias, X]
        self.m = final.shape[1]
        return final

    def _init_W_h(self, shape):
        bias = np.ones((shape[0], 1))
        weights = self.rs.normal(size=shape)
        final = np.c_[bias, weights]
        self.d = final.shape[1]
        return final
    
    def onehot(self, y):
        y = y.reshape(-1, 1)
        return oh_encoder.fit_transform(y)
    
    def sigmoid(self, Z):
        '''
        implementing sigmoid function
        '''
        return 1 / (1 + np.exp(Z))
    
    def softmax(self, Z):
        '''
        implementing softmax function
        '''
        denom = np.exp(Z)
        return denom / denum.sum(axis=0) # sum by row
        
    def predict(self, X):
        '''
        return a label
        '''
        ypredict = self.forward(X)
        return np.argmax(ypredict, axis=1)