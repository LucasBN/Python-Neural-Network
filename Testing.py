import NeuralNetwork
import numpy as np

# Input data x = [Hours of sleep, hours of studing] and y = [Test score]:
x = np.array(([0,0], [10,10], [5,5], [7.5, 7.5]), dtype=float)
y = np.array(([0], [100], [50], [75]), dtype=float)

# Normalize
x = x/np.amax(x)
y = y/100 # Max test score is 100

# [Hours of sleep, hours of studying]
dataset = np.array(([0,0], [10,10], [5,5], [7.5, 7.5]), dtype=float)

# Normalize
dataset = dataset/np.amax(dataset, axis=0) 

# Train the neural network with input data:
NeuralNetwork.T.train(x, y)

# Make prediction based on the training:
print(NeuralNetwork.NN.forward(dataset))




