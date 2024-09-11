'''
PINEDA, BRIXTER SIEN M.
2020-2019
CMSC 191 - NEURAL NETWORKS

A Neural Network Predicting the Result of a Shop's Advertisement

'''

# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functions import *

# Neural Network hyperparameters
epochs = 1000
learnrate = 0.001

# Reading the csv file into a pandas DataFrame
data = pd.read_csv('shopAdvertisement_data.csv', sep='\t')
data.head()

data_train, data_val, data_test = processData(data)

########## Splitting the Train data into Features (X) and Targets (Y) ##########

# List of target columns 
target_columns = ['Response', 'AcceptedCmp3', 'AcceptedCmp4',	'AcceptedCmp5',	'AcceptedCmp1',	'AcceptedCmp2']

# Splitting features and targets for training data
X_train = data_train.drop(target_columns, axis=1)  # Drop the target columns to get the features/inputs
Y_train = data_train[target_columns]  # Select the target columns/outputs

# Splitting features and targets for validating data
X_val = data_val.drop(target_columns, axis=1)  # Drop the target columns to get the features/inputs
Y_val = data_val[target_columns]  # Select the target columns/outputs

# Splitting features and targets for test data
X_test = data_test.drop(target_columns, axis=1)  # Drop the target columns to get the features
Y_test = data_test[target_columns]  # Select the target columns



print("\n--------------- START TRAINING -----------------------\n")

# Define input size, hidden layer size, and output size
input_size = X_train.shape[1]
hidden_size = 5  # Arbitrary hidden layer size
output_size = Y_train.shape[1]

# Create and train the neural network
nn = NeuralNetwork(input_size, hidden_size, output_size, learnrate)
nn.train(X_train, Y_train, X_val, Y_val, epochs)

# print("X_train shape:", X_train.shape)
# print("y_train shape:", Y_train.shape)
# print("X_val shape:", X_val.shape)
# print("y_val shape:", Y_val.shape)
# print("X_test shape:", X_test.shape)
# print("y_test shape:", Y_test.shape)

print("--------------- DONE TRAINING -----------------------\n")

# Test the model on the test set
nn.test(X_test, Y_test)