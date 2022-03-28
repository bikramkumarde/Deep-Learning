import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

#This is the forward propagation step of the hidden layer. Here W, b and the data is received as input. The operation W.X + b is performed after taking appropriate transpose. The final value is then passed through a ReLU function. Then the activated value is multiplied with a Dropout matrix
def forward_prop_hidden_Drop(W, b, data_X, D, keep_prob):    
    intermediate = np.dot(W, data_X)
    Z = intermediate + b
    A = tanh(Z)
    A = A*D
    A = A/keep_prob
    return A,Z
def forward_prop_hidden(W, b, data_X):     # This is the forward propagation step of the hidden layer. Here W, b and the data is received as input. The operation W.X + b is performed after taking appropriate transpose. The final value is then passed through a ReLU function
	intermediate = np.dot(W, data_X)
	Z = intermediate + b
	A = tanh(Z)
	return A,Z
def forward_prop_output(W, b, data_X):        #This is the forward propagation step of the output layer. Here W, b and the data is received as input. The operation W.X + b is performed after taking appropriate transpose. The final value is then passed through a sigmoid function to make the value between 0 - 1
    intermediate = np.dot(W, data_X)
    Z = intermediate + b
    A = sigmoid(Z)
    return A
#This is the backward propagation step. Here the required derivatives are calculated which will be later updated in further steps. Also the dropout matrix is incorporated and there is a separate function where dropout matrix is not present which will be required for calculating accuracy. The subsequent functions are used for unregularized neural network and neural net with L2 regularization respectively.
def backward_prop_1_Drop(A1, m, W2, dZ2, Z1, data_X, D, keep_prob):
    dA = np.dot(W2.T, dZ2)
    dA = dA*D
    dA = dA/keep_prob
    dZ1 = dA* tanh_derivative(Z1)
    dW1 = np.dot(dZ1, data_X.T)/m
    db1 = np.sum(dZ1, axis =1)/m
    db1 = db1.reshape(-1, 1)
    return dW1, db1, dZ1
def backward_prop_1(A1, m, W2, dZ2, Z1, data_X):
    dZ1 = np.dot(W2.T, dZ2) 
    dZ1 = dZ1* tanh_derivative(Z1)
    dW1 = np.dot(dZ1, data_X.T)/m
    db1 = np.sum(dZ1, axis =1)/m
    db1 = db1.reshape(-1, 1)
    return dW1, db1, dZ1
def backward_prop_1_L2(A1, m, W2, dZ2, Z1, data_X, lambda1, W1):
    dZ1 = np.dot(W2.T, dZ2) 
    dZ1 = dZ1* tanh_derivative(Z1)
    dW1 = (np.dot(dZ1, data_X.T)+lambda1*W1)/m
    db1 = np.sum(dZ1, axis =1)/m
    db1 = db1.reshape(-1, 1)
    return dW1, db1, dZ1
def backward_prop_2(A2, A1, m, data_Y):   #This is the backward propagation of output layer(Layer2)
    dZ2 = A2 - data_Y
    dW2 = np.dot(dZ2, A1.T)/m
    db2 = np.sum(dZ2, axis =1)/m
    db2 = db2.reshape(-1, 1)
    return dW2, db2, dZ2
def sigmoid(x):                         #This is the sigmoid activation function used to map any real value between 0 and 1
    ans = 1 / (1 + np.exp(-x))
    return ans
def relu(x):							#This is the Rectified Linear Unit(ReLU) activation function
	ans = np.maximum(0,x)
	return ans
def relu_derivative(x):					#This is the derivative of the ReLU function
    y = np.array(x > 0)
    y = 1*y
    return y                
def tanh(x):							#This is the tanh activation function
	ans = np.tanh(x)
	return ans
def tanh_derivative(x):                 #This is the derivative of tanh function
	ans = 1/np.square(np.cosh(x))
	return ans
def cost_function(A, m, data_Y):     #Computes the cost function for all the training samples
    total_cost = -(1 / m) * np.sum(data_Y * np.log(A) + (1 - data_Y) * np.log(1 - A))
    return total_cost            
def cost_function_L2(A, m, data_Y, W, lambda1, l):     #This is cost function for L2 regularization
    total_cost = -(1 / m) * np.sum(data_Y * np.log(A) + (1 - data_Y) * np.log(1 - A))
    sum1 = 0
    for k in range(l-1):
        sum1 = sum1 + np.sum(np.square(W[k]))
    total_cost = total_cost + sum1*lambda1/(2*m)
    return total_cost
def results(W, b, data_X, data_Y, l):        #Here the results are predicted. The W and b of model are taken as input. The values are predicted using test dataset and then using the actual values the accuracy is computed
    A1 = data_X
    for k in range(l-2):
        A1, Z1 = forward_prop_hidden(W[k], b[k], A1)        
    A2 =  forward_prop_output(W[l-2], b[l-2], A1)   
    pred_Y = A2.transpose()
    
    pred_Y = np.around(pred_Y, decimals = 0)   #Rounding off the predicted value to 0 or 1    
    accuracy = accuracy_score(data_Y.T, pred_Y)*100
    precision = precision_score(data_Y.T, pred_Y)*100
    recall = recall_score(data_Y.T, pred_Y)*100
    fscore = f1_score(data_Y.T, pred_Y)*100
    return accuracy, precision, recall, fscore

#Plotting the cost function as a function of the iterations. In general we can see that the cost function decreases very fast initially but after a certain number of iterations the rate of decrease is much less which indicates that our neural network is converging hence the error is reducing asymptotically.
def cost_graph(J_arr, cost_arr):
    plt.title('Cost Function Graph')
    plt.xlabel('Epoch')
    plt.ylabel('Cost Function')
    plt.plot(J_arr, cost_arr)
    plt.show()        