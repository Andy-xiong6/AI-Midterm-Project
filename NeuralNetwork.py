import numpy as np

class MLP:
    class Neuron:
        def __init__(self, n_input, n_output, activation):
            self.n_input = n_input # number of input neurons
            self.n_output = n_output # number of output neurons
            
            if activation == 'sigmoid':
                self.activation = MLP.functions.sigmoid
            elif activation == 'tanh':
                self.activation = MLP.functions.tanh
            elif activation == 'relu':
                self.activation = MLP.functions.relu
                
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
        
        def momentum_update(self, dw, db, alpha, gama, vdw, vdb):
            vdw = gama * vdw + alpha * dw
            vdb = gama * vdb + alpha * db
            self.weights -= alpha * vdw
            self.bias -= alpha * vdb
    
    class Layer:
        def __init__(self, n_input, n_output, activation, momentum=False, OutputMode=False):
            self.n_input = n_input
            self.n_output = n_output
            
            if activation == 'sigmoid':
                self.activation = MLP.functions.sigmoid
            elif activation == 'tanh':
                self.activation = MLP.functions.tanh
            elif activation == 'relu':
                self.activation = MLP.functions.relu

            self.neurons = []
            for i in range(n_output):
                neuron = MLP.Neuron(n_input, n_output, activation)
                self.neurons.append(neuron)
                
            if OutputMode:
                self.cost = MLP.functions.cross_entropy
            else:
                self.cost = MLP.functions.MSE
            
        def forward(self, x):
            self.x = x
            a = []
            
            for neuron in self.neurons:
                output = neuron.forward(x)
                a.append(output)

            self.a = np.array(a).T
            return self.a 
        
        def backward(self, da):
            dx = []
            dw = []
            db = []
            
            for i in range(len(self.neurons)):
                gradient = self.neurons[i].backward(da[:, i])
                dx.append(gradient[0])
                dw.append(gradient[1])
                db.append(gradient[2])
                
            dx = np.sum(dx, axis=0)
            dw = np.sum(dw, axis=0)
            db = np.sum(db, axis=0)
        
            return dx, dw, db
        
        def update(self, dw, db, alpha):
            for neuron in self.neurons:
                neuron.update(dw, db, alpha)
                
        def momentum_update(self, dw, db, alpha, gama, vdw, vdb):
            for neuron in self.neurons:
                neuron.momentum_update(dw, db, alpha, gama, vdw, vdb)
    class functions:
        def sigmoid(self, z , derivative=False):
            if derivative:
                return self.sigmoid(z) * (1 - self.sigmoid(z))
            return 1 / (1 + np.exp(-z))
        
        
        def tanh(self, z, derivative=False):
            if derivative:
                return 1 - np.square(np.tanh(z))
            return np.tanh(z)
        
        def relu(self, z, derivative=False):
            if derivative:
                return self.d_relu(z)
            return np.maximum(0, z)
        
        def MSE(self, y, y_hat, derivative=False):
            if derivative:
                return self.d_MSE(y, y_hat)
            return np.mean(np.square(y - y_hat))
        
        def cross_entropy(self, y, y_hat, derivative=False):
            if derivative:
                return self.d_cross_entropy(y, y_hat)
            return -np.sum(y * np.log(y_hat))
        
    def __init__(self, n_input, n_output, hidden_number, activations):
        # n_input: number of input neurons
        # n_output: number of output neurons
        # hidden_number: list of hidden layer neurons, it should be a list of integers
        # activations: list of activation functions, it should be strings including 'sigmoid', 'tanh', 'relu'
        self.n_input = n_input
        self.n_output = n_output
        self.hidden_number = hidden_number
        for i in range(len(hidden_number)):
            activation = activations[i]
            if i == 0:
                layer = MLP.Layer(n_input, hidden_number[i], activation)
            elif i != len(hidden_number) - 1:
                layer = MLP.Layer(hidden_number[i-1], hidden_number[i], activation)
            else:
                layer = MLP.Layer(hidden_number[i], n_output, activation , OutputMode=True)
            self.layers.append(layer)
        
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
        for i in range(epochs):
            y_hat = self.forward(x)
            self.backward(y, y_hat)
    
    def predict(self, x):
        return self.forward(x)
            

    
    
            