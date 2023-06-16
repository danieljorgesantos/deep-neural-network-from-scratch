'''
-------------------------------------------------------------------------------------------------------------------------------
IMPORTS AND DEFINITIONS
-------------------------------------------------------------------------------------------------------------------------------
'''
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    sigmoid_x = sigmoid(x)
    return sigmoid_x * (1 - sigmoid_x)

def swish(x, beta=1):
    return x * (1 / (1 + np.exp(-beta * x)))

def swish_derivative(x, beta=1):
    return swish(x, beta) + sigmoid(beta * x) * (1 - swish(x, beta))

def calculate_loss(y_true, y_pred):
    epsilon = 1e-7  # small constant to avoid division by zero
    loss = -np.mean(y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon))
    return loss

'''
-------------------------------------------------------------------------------------------------------------------------------
DATASET, XOR Problem
-------------------------------------------------------------------------------------------------------------------------------
'''
# Inputs
neural_network_input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# Real results
neural_network_real_result = np.array([[0], [1], [1], [0]])

'''
-------------------------------------------------------------------------------------------------------------------------------
NEURAL NET TRAINING AND BACKPROPAGATION
-------------------------------------------------------------------------------------------------------------------------------
'''

# Randomly initialize the weights and biases
np.random.seed(0)
w_0 = 2 * np.random.random((neural_network_input.shape[1], 120)) - 1
b_0 = 2 * np.random.random((1, 120)) - 1

w_1 = 2 * np.random.random((120, 120)) - 1
b_1 = 2 * np.random.random((1, 120)) - 1

w_2 = 2 * np.random.random((120, 120)) - 1
b_2 = 2 * np.random.random((1, 120)) - 1

w_3 = 2 * np.random.random((120, neural_network_real_result.shape[1])) - 1
b_3 = 2 * np.random.random((1, neural_network_real_result.shape[1])) - 1

epochs = 100000

learning_rate = 0.01

for epoch in range(epochs):
    
    # Forward propagation
    layer_0 = neural_network_input
    layer_1 = sigmoid(np.dot(layer_0, w_0) + b_0)
    layer_2 = sigmoid(np.dot(layer_1, w_1) + b_1)
    layer_3 = sigmoid(np.dot(layer_2, w_2) + b_2)
    layer_4 = sigmoid(np.dot(layer_3, w_3) + b_3)

    # Calculate the error and Backpropagation
    layer_4_error = neural_network_real_result - layer_4
    layer_4_delta = layer_4_error * sigmoid_derivative(layer_4)

    layer_3_error = layer_4_delta.dot(w_3.T)
    layer_3_delta = layer_3_error * sigmoid_derivative(layer_3)

    layer_2_error = layer_3_delta.dot(w_2.T)
    layer_2_delta = layer_2_error * sigmoid_derivative(layer_2)

    layer_1_error = layer_2_delta.dot(w_1.T)
    layer_1_delta = layer_1_error * sigmoid_derivative(layer_1)

    # Update the weights and biases
    w_3 += learning_rate * layer_3.T.dot(layer_4_delta)
    b_3 += learning_rate * np.sum(layer_4_delta, axis=0, keepdims=True)

    w_2 += learning_rate * layer_2.T.dot(layer_3_delta)
    b_2 += learning_rate * np.sum(layer_3_delta, axis=0, keepdims=True)

    w_1 += learning_rate * layer_1.T.dot(layer_2_delta)
    b_1 += learning_rate * np.sum(layer_2_delta, axis=0, keepdims=True)

    w_0 += learning_rate * layer_0.T.dot(layer_1_delta)
    b_0 += learning_rate * np.sum(layer_1_delta, axis=0, keepdims=True)

    # print the loss and epoch to see training on screen
    if epoch % 100 == 0:
        loss = calculate_loss(neural_network_real_result, layer_4)
        print(f"Epoch: {epoch+1}, Loss: {loss}")

'''
-------------------------------------------------------------------------------------------------------------------------------
PREDICTING WITH THE NEURAL NET
-------------------------------------------------------------------------------------------------------------------------------
'''

# Set the print options to suppress the exponent
np.set_printoptions(suppress=True)

neural_network_input = np.array([[1, 0]])

# Forward propagation
layer_0 = neural_network_input
layer_1 = sigmoid(np.dot(layer_0, w_0) + b_0)
layer_2 = sigmoid(np.dot(layer_1, w_1) + b_1)
layer_3 = sigmoid(np.dot(layer_2, w_2) + b_2)
layer_4 = sigmoid(np.dot(layer_3, w_3) + b_3)

print('')
print('layer_4 result:', layer_4)