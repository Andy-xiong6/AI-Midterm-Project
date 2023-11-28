import numpy as np

class MCLogisticRegression:
    def __init__(self, n, m, k):
        # n represents the number of examples
        # m represents the dimension of input
        # k represents the number of classes
        self.n = n
        self.m = m
        self.k = k
        self.W = np.random.rand(m, k)
        
    def linearmodel(self, X):
        return X @ self.W  # nxk
    
    def softmax(self, X):
        exp_X = np.exp(X - np.max(X, axis=1, keepdims=True))  #subtract maximum value
        return exp_X / np.sum(exp_X, axis=1, keepdims=True)
    
    def gradient(self, X, Y):
        # X nxm 
        # Y nxk
        # W mxk 
        error = self.softmax(self.linearmodel(X)) - Y # nxk
        gradient = X.T @ error
        return gradient
    
    def preprocess(self, X):
        return np.insert(X, 0, 1, axis=1) # insert 1 to the first column
    
    def one_hot_encoding(self, Y, num_classes):
        return np.eye(num_classes)[Y.reshape(-1)]

    def train(self, X, Y, epoch, lamda):
        for i in range(0, epoch):
            delta_W = self.gradient(X, Y)
            self.W -= lamda * delta_W # gradient descent
        return self.W
    
    
    
    