from LogisticRegression import MCLogisticRegression
from NeuralNetwork import MLP
import numpy as np
'''
LogisticRegression on binary classification
'''
#import data
train_data = np.loadtxt('train.data', delimiter=',')
train_input = train_data[:,0:2]
train_label = train_data[:,2]

test_data = np.loadtxt('test.data', delimiter=',')
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

'''
Multi-layer Perceptron with single hidden layer but different numbers of neurons
'''
#import data
train_data = np.loadtxt('train.data', delimiter=',')
train_input = train_data[:,0:2].T
train_label = train_data[:,2]

test_data = np.loadtxt('test.data', delimiter=',')
test_input = test_data[:,0:2].T
test_label = test_data[:,2]


def one_hidden_layer_train(activation_function, epoch, learning_rate):
    hidden_numbers = []
    activations = []
    for i in range(1,21):
        hidden_numbers.clear()
        hidden_numbers.append(i)
        activations.append(activation_function)
        mlp = MLP(len(train_input), len(train_label), hidden_numbers, activations)
        mlp.train(train_input, train_label, epoch, learning_rate)
        mlp.predict(test_input)
        accuracy = np.sum(np.argmax(mlp.y_hat, axis=1) == np.argmax(test_label, axis=1)) / len(test_data)
        print("accuracy: ", accuracy)
    
#using sigmoid as the activation function
one_hidden_layer_train('sigmoid', 1000, 0.1)
#using tanh as the activation function
one_hidden_layer_train('tanh', 1000, 0.1)
#using relu as the activation functino
one_hidden_layer_train('relu', 1000, 0.1)

