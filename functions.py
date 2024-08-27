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