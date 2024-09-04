import numpy as np
from typing import List, Literal, Union

class NNetwork:
    def __init__(self, layers: List[int], activation: Literal["relu", "sigmoid"] = "relu", 
                 optimizer: Literal["sgd", "adam"] = "sgd", learning_rate: float = 0.01) -> None:
        self.parameters = {}
        self.A_cache = {}
        self.Z_cache = {}
        self.num_layers = len(layers)
        self.activation = activation
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.num_classes = layers[-1]
        assert self.num_layers > 1, "Network must have at least two layers"

        # Initialize parameters (weights and biases)
        for l in range(1, self.num_layers):
            self.parameters['W' + str(l)] = np.random.randn(layers[l], layers[l - 1]) * 0.01
            self.parameters['b' + str(l)] = np.zeros((layers[l], 1))

        # Adam optimizer initialization
        if optimizer == "adam":
            self.v = {k: np.zeros_like(v) for k, v in self.parameters.items()}
            self.s = {k: np.zeros_like(v) for k, v in self.parameters.items()}
            self.t = 0

    def empty_cache(self) -> None:
        self.A_cache = {}
        self.Z_cache = {}

    def _relu(self, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        A = np.maximum(0, Z)
        return A, Z

    def _sigmoid(self, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        A = 1 / (1 + np.exp(-Z))
        return A, Z
    
    def _softmax(self, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))  # Stability fix
        A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
        return A, Z
    
    def _relu_back(self, dA: np.ndarray, Z: np.ndarray) -> np.ndarray:
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        return dZ
    
    def _sigmoid_back(self, dA: np.ndarray, Z: np.ndarray) -> np.ndarray:
        s = 1 / (1 + np.exp(-Z))
        return dA * s * (1 - s)
    
    def _softmax_back(self, A: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return A - Y  # Softmax + CE combined gradient
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        self.empty_cache()
        self.A_cache['A0'] = X

        for l in range(1, self.num_layers):
            Z = np.dot(self.parameters[f'W{l}'], self.A_cache[f'A{l-1}']) + self.parameters[f'b{l}']
            self.Z_cache[f'Z{l}'] = Z
            
            if l == self.num_layers - 1:  # Output layer
                if self.num_classes == 1:
                    A, _ = self._sigmoid(Z)
                else:
                    A, _ = self._softmax(Z)
            else:
                A, _ = self._relu(Z) if self.activation == "relu" else self._sigmoid(Z)
            
            self.A_cache[f'A{l}'] = A

        return self.A_cache[f'A{self.num_layers-1}']

    def backward(self, Y: np.ndarray) -> dict:
        m = Y.shape[1]
        grads = {}
        
        # Output layer
        A_last = self.A_cache[f'A{self.num_layers-1}']
        if self.num_classes == 1:
            dA = - (np.divide(Y, A_last) - np.divide(1 - Y, 1 - A_last))
            dZ = self._sigmoid_back(dA, self.Z_cache[f'Z{self.num_layers-1}'])
        else:
            dZ = self._softmax_back(A_last, Y)

        # Backpropagate through the network
        for l in reversed(range(1, self.num_layers)):
            A_prev = self.A_cache[f'A{l-1}']
            grads[f'dW{l}'] = np.dot(dZ, A_prev.T) / m
            grads[f'db{l}'] = np.sum(dZ, axis=1, keepdims=True) / m
            
            if l > 1:
                dA = np.dot(self.parameters[f'W{l}'].T, dZ)
                Z = self.Z_cache[f'Z{l-1}']
                dZ = self._relu_back(dA, Z) if self.activation == "relu" else self._sigmoid_back(dA, Z)

        return grads

    def optimizer_step(self, grads: dict) -> None:
        if self.optimizer == "sgd":
            for l in range(1, self.num_layers):
                self.parameters[f'W{l}'] -= self.learning_rate * grads[f'dW{l}']
                self.parameters[f'b{l}'] -= self.learning_rate * grads[f'db{l}']
        elif self.optimizer == "adam":
            beta1, beta2 = 0.9, 0.999
            epsilon = 1e-8
            self.t += 1
            
            for l in range(1, self.num_layers):
                for param in ['W', 'b']:
                    self.v[f'{param}{l}'] = beta1 * self.v[f'{param}{l}'] + (1 - beta1) * grads[f'd{param}{l}']
                    self.s[f'{param}{l}'] = beta2 * self.s[f'{param}{l}'] + (1 - beta2) * (grads[f'd{param}{l}']**2)
                    
                    v_corrected = self.v[f'{param}{l}'] / (1 - beta1**self.t)
                    s_corrected = self.s[f'{param}{l}'] / (1 - beta2**self.t)
                    
                    self.parameters[f'{param}{l}'] -= self.learning_rate * v_corrected / (np.sqrt(s_corrected) + epsilon)

    def calculate_loss(self, Y: np.ndarray, A: np.ndarray) -> float:
        m = Y.shape[1]
        if self.num_classes == 1:
            # Binary Cross-Entropy Loss
            loss = -np.mean(Y * np.log(A) + (1 - Y) * np.log(1 - A))
        else:
            # Categorical Cross-Entropy Loss
            loss = -np.mean(np.sum(Y * np.log(A), axis=0))
        return loss

    def predict(self, X: np.ndarray) -> np.ndarray:
        A = self.forward(X)
        if self.num_classes == 1:
            return (A > 0.5).astype(int)
        else:
            return np.argmax(A, axis=0)

    def train(self, X: np.ndarray, Y: np.ndarray, epochs: int, batch_size: int = 32) -> List[float]:
        m = X.shape[1]
        losses = []

        for epoch in range(epochs):
            epoch_loss = 0
            
            for i in range(0, m, batch_size):
                X_batch = X[:, i:i+batch_size]
                Y_batch = Y[:, i:i+batch_size]
                
                A = self.forward(X_batch)
                grads = self.backward(Y_batch)
                self.optimizer_step(grads)
                
                batch_loss = self.calculate_loss(Y_batch, A)
                epoch_loss += batch_loss * X_batch.shape[1]
            
            epoch_loss /= m
            losses.append(epoch_loss)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {epoch_loss}")
        
        return losses
