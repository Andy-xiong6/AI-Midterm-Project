from LogisticRegression import MCLogisticRegression
from NeuralNetwork import MLP
import numpy as np

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