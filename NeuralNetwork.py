import numpy as np

class MLP:
    class Neruon:
        def __init__(self, n_input, n_output, activation):
            self.n_input = n_input # number of input neurons
            self.n_output = n_output # number of output neurons
            self.activation = activation # activation function
            self.weights = np.random.rand(n_input, n_output)
            self.bias = np.random.rand(n_output)

        def forward(self, x):
            self.x = x
            self.z = np.dot(x, self.weights) + self.bias # z = w * x + b
            self.a = self.activation(self.z) # a = f(z)
            return self.a
        
        def backward(self, da):
            dz = da * self.activation(self.z, derivative=True) # dz = da * f'(z)
            dw = np.dot(self.x.T, dz) # dw = x.T * dz
            db = np.sum(dz, axis=0) # db = sum(dz)
            dx = np.dot(dz, self.weights.T) # dx = dz * w.T
            return dx, dw, db
        
        def update(self, dw, db, alpha):
            self.weights -= alpha * dw
            self.bias -= alpha * db
    
    class Layer:
        def __init__(self, n_input, n_output, activation):
            self.n_input = n_input
            self.n_output = n_output
            self.activation = activation
            self.neurons = [MLP.Neruon(n_input, n_output, activation) for _ in range(n_output)]
        
        def forward(self, x):
            self.x = x
            self.a = np.array([neuron.forward(x) for neuron in self.neurons]).T
            return self.a
        
        def backward(self, da):
            dx = np.array([neuron.backward(da[:,i]) for i, neuron in enumerate(self.neurons)])
            dx = np.sum(dx[:,0], axis=0)
            dw = np.sum(dx[:,1], axis=0)
            db = np.sum(dx[:,2], axis=0)
            return dx, dw, db
        
        def update(self, dw, db, alpha):
            for neuron in self.neurons:
                neuron.update(dw, db, alpha)
        
    def __init__(self, n_input, n_output, n_hidden, activation):
        self.n_input = n_input
        self.n_output = n_output
        self.n_hidden = n_hidden
        self.activation = activation
        self.layers = [MLP.Layer(n_input, n_hidden, activation), MLP.Layer(n_hidden, n_output, activation)]
        
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, y, y_hat):
        dy = y - y_hat
        for layer in reversed(self.layers):
            dy, dw, db = layer.backward(dy)
            layer.update(dw, db, 0.01)
    
    def train(self, x, y, epochs):
        for _ in range(epochs):
            y_hat = self.forward(x)
            self.backward(y, y_hat)
    
    def predict(self, x):
        return self.forward(x)
            
    def sigmoid(self,z):
        return 1 / (1 + np.exp(-z))
    
    def d_sigmoid(self,z):
        return self.sigmoid(z) * (1 - self.sigmoid(z)) # derivative of sigmoid
            