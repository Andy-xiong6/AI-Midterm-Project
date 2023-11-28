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

if __name__ == '__main__':
    #import data
    train_data = np.loadtxt('midterm/train.data', delimiter=',')
    train_input = train_data[:,0:2]
    train_label = train_data[:,2]
    
    test_data = np.loadtxt('midterm/test.data', delimiter=',')
    test_input = test_data[:,0:2]
    test_label = test_data[:,2]
    
    classes_number = np.unique(train_label).size
    LR = MCLogisticRegression(len(train_data), 3, classes_number)
    
    #preprocess data
    train_input = LR.preprocess(train_input)
    test_input = LR.preprocess(test_input)
    
    train_label = LR.one_hot_encoding(train_label.astype(int), classes_number)
    test_label = LR.one_hot_encoding(test_label.astype(int), classes_number)
    
    #train
    LR.train(train_input, train_label, 1000, 0.1)
    
    #predict
    y_hat = LR.one_hot_encoding(np.argmax(LR.softmax(LR.linearmodel(test_input)), axis=1), classes_number)
    print(y_hat)
    #accuracy
    accuracy = np.sum(np.argmax(y_hat, axis=1) == np.argmax(test_label, axis=1)) / len(test_data)
    print("accuracy: ", accuracy)
    
    
    
    