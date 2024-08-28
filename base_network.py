import numpy as np
from typing import Any, List


class NNetwork:
    def __init__(self, layers: List, activation="relu", optimizer="sgd") -> np.ndarray:
        self.parameters = {}
        self.A_cache = {}
        self.Z_cache = {}
        self.num_layers = len(layers)
        assert self.num_layers > 0

        for l in range(1, self.num_layers):
            self.parameters['W' + str(l)] = np.random.randn(layers[l], layers[l - 1]) 
            self.parameters['b' + str(l)] = np.zeros((layers[l], 1))

    def empty_cache(self):
        self.A_cache = {}
        self.Z_cache = {}

    def _relu(self, Z):
        A = np.maximum(0,Z)
        cache = Z 
        return A, cache

    def _sigmoid(self, Z):
         A = 1/(1+np.exp(-Z))
         cache = Z
         return A, cache
    
    def _relu_back(self, dA, cache):
        Z = cache
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0

        assert (dZ.shape == Z.shape)

        return dZ
    
    def _sigmoid_back(self, dA, cache):
        Z = cache

        s = 1/(1+np.exp(-Z))
        dZ = dA * s * (1-s)

        assert (dZ.shape == Z.shape)

        return dZ
    
    def forward(self, X):
        self.empty_cache()
        self.A_cache['A' + str(0)] = X

        for l in range(1, self.num_layers + 1):
            self.Z_cache['Z' + str(l)] = np.dot(self.parameters['W' + str(l)], self.A_cache['A' + str(l - 1)]) + (self.parameters['b' + str(l)])
            self.A_cache['A' + str(l)] = self._relu(self.Z_cache['Z' + str(l)])

        return self.A_cache['A' + str(self.num_layers)]

    def backward(self):
        m = (self.A_cache['A'  + str(self.num_layers)]).shape[1]
        dA = {}
        dZ = {}
        dW = {}
        db = {}

    def optimizer_step(self):
        raise NotImplementedError
    
    def __call__(self, X):
        return self.predict(X)
    
    def predict(self, X):
        raise NotImplementedError
