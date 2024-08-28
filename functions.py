import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def convert_to_timestamp(date_column):
    # Convert the date column to datetime objects using the specified format
    date_column = pd.to_datetime(date_column, format="%d-%m-%Y")
    
    # Convert datetime to Unix timestamp (seconds since the Unix epoch)
    timestamp_column = date_column.apply(lambda x: x.timestamp())
    
    return timestamp_column


def normalizeData(data, type = 0):
    min = data.min() if type else 0 
    max = data.max()
    return (data - min) / (max - min)

############ FUNCTIONS FOR TRAINING 1-LAYER NEURAL NETWORK ############
# Activation (sigmoid) function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid(x)
def sigmoid_prime(x):
    return sigmoid(x) * (1-sigmoid(x))

def error_formula(y, output):
    return - y*np.log(output) - (1 - y) * np.log(1-output)

def error_term_formula(x, y, output):
    return (y - output)*x


################### PLOTTING DATA #############################

# TODO: Complete code
def plot_points(data):
    #store columns Year_Birth, Education, Marital_Status, and Income to X
    X = np.array(data[["Year_Birth", "Education", "Marital_Status", "Income"]]) 
    y = np.array(data["Response"]) #outputs
    response_accept = X[np.argwhere(y==1)]
    response_reject = X[np.argwhere(y==0)]
    plt.scatter([s[0][0] for s in response_reject], [s[0][1] for s in response_reject], s = 25, color = 'red', edgecolor = 'k')
    plt.scatter([s[0][0] for s in response_accept], [s[0][1] for s in response_accept], s = 25, color = 'cyan', edgecolor = 'k')





################### Neural Network ###################

# Training function
def train_nn(features, targets, epochs, learnrate):
    
    # Use to same seed to make debugging easier
    np.random.seed(42)
    n_records, n_features = features.shape
    last_loss = None

    # Initialize weights
    weights = np.random.normal(scale=1 / n_features**.5, size=n_features)

    for e in range(epochs):
        prev_w = np.zeros(weights.shape)
        for x, y in zip(features.values, targets):
            # Loop through all records, x is the input, y is the target

            # Activation of the output unit
            #   Notice we multiply the inputs and the weights here 
            #   rather than storing h as a separate variable 
            output = sigmoid(np.dot(x, weights))

            # The error term
            error_term = error_term_formula(x, y, output)

            # The gradient descent step, the error times the gradient times the inputs
            
            prev_w += np.array(error_term, dtype=float)

        # Update the weights here. The learning rate times the 
        # change in weights
        # don't have to divide by n_records since it is compensated by the learning rate
        weights += learnrate * prev_w #/ n_records  

        # Printing out the mean square error on the training set
        if e % (epochs / 10) == 0:
            x1 = np.dot(features, weights)
            # print(weights.dtype)
            out = sigmoid(x1.astype(float))
            loss = np.mean(error_formula(targets, out))
            print("Epoch:", e)
            if last_loss and last_loss < loss:
                print("Train loss: ", loss, "  WARNING - Loss Increasing")
            else:
                print("Train loss: ", loss)
            last_loss = loss
            print("=========")
    print("Finished training!")
    return weights
    
