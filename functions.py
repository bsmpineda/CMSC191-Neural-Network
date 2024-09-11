import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from datetime import datetime


# Neural Network Class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        # Initialize weights and biases
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Weights between input and hidden layers
        self.weights_input_hidden = np.random.rand(self.input_size, self.hidden_size)
        self.bias_hidden = np.random.rand(self.hidden_size)

        # Weights between hidden and output layers
        self.weights_hidden_output = np.random.rand(self.hidden_size, self.output_size)
        self.bias_output = np.random.rand(self.output_size)

    def feedforward(self, X):
        # Feedforward propagation
        self.bias_hidden = np.array(self.bias_hidden)
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = sigmoid(self.hidden_input)

        self.bias_output = np.array(self.bias_output)
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.output_output = sigmoid(self.output_input)
    
        return self.output_output

    def backpropagate(self, X, y, output):
        # Backpropagation
        error = y - output  # Error at output layer

        # Gradient for output layer
        d_output = error * sigmoid_derivative(output)

        # Error at hidden layer
        error_hidden = d_output.dot(self.weights_hidden_output.T)
        d_hidden = error_hidden * sigmoid_derivative(self.hidden_output)

        # Update weights and biases
        self.weights_hidden_output += self.hidden_output.T.dot(d_output) * self.learning_rate
        self.bias_output += np.sum(d_output, axis=0) * self.learning_rate

        self.weights_input_hidden += X.T.dot(d_hidden) * self.learning_rate
        self.bias_hidden += np.sum(d_hidden, axis=0) * self.learning_rate

    def train(self, X_train, y_train, X_val, y_val, epochs=1000, patience=20):
        best_val_loss = np.inf
        patience_counter = 0
        best_weights = None

        for epoch in range(epochs):
            # Forward pass for training data
            output = self.feedforward(X_train)
            self.backpropagate(X_train, y_train, output)

            # Calculate training and validation loss
            train_loss = mse_loss(y_train, output)
            val_output = self.feedforward(X_val)
            val_loss = mse_loss(y_val, val_output)
            val_acc = accuracy(y_val, val_output)

            # Print progress
            if (epoch + 1) % 100 == 0:
                print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%')

            # Early stopping logic
            if round(val_loss, 6) < round(best_val_loss, 6):
                best_val_loss = val_loss
                best_weights = (self.weights_input_hidden.copy(), self.bias_hidden.copy(),
                                self.weights_hidden_output.copy(), self.bias_output.copy())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

        # Restore best weights
        if best_weights:
            self.weights_input_hidden, self.bias_hidden, self.weights_hidden_output, self.bias_output = best_weights
            print("Restored best model weights based on validation loss.")

    def test(self, X_test, y_test):
        # Perform testing by predicting on test set
        predictions = self.feedforward(X_test)
        acc = accuracy(y_test, predictions)
        print(f"Test Accuracy: {acc:.2f}%")
        return acc


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

def processData(data):
        # Drop ID, ID is not included
    data = data.drop("ID", axis="columns") 
    print("Done Loading the data...\n")

    ############# ONE-HOT ENCODING THE RANK ################
    # Make dummy variables for Education
    one_hot_data = pd.concat([data, pd.get_dummies(data['Education'], prefix='Education')], axis=1)

    # Drop the previous Education column
    one_hot_data = one_hot_data.drop("Education", axis='columns')

    # Make dummy variables for Marital Status
    one_hot_data = pd.concat([one_hot_data, pd.get_dummies(one_hot_data['Marital_Status'], prefix='MStat')], axis=1)

    # Drop the previous Marital Status column
    one_hot_data = one_hot_data.drop("Marital_Status", axis='columns')

    # Make dummy variables for Marital Status
    one_hot_data = pd.concat([one_hot_data, pd.get_dummies(one_hot_data['Kidhome'], prefix='Kidhome')], axis=1)

    # Drop the previous Marital Status column
    one_hot_data = one_hot_data.drop("Kidhome", axis='columns')

    # Make dummy variables for Marital Status
    one_hot_data = pd.concat([one_hot_data, pd.get_dummies(one_hot_data['Teenhome'], prefix='Teenhome')], axis=1)

    # Drop the previous Marital Status column
    one_hot_data = one_hot_data.drop("Teenhome", axis='columns')

    print("Done one hot encoding...\n")

    ###############  NORMALIZATION and SCALING DATA  ################
    # THIS IS TO MAKE SURE THAT DATA ARE IN [0, 1]

    # Making a copy of our data
    processed_data = one_hot_data[:]

    # Scale the columns

    # Columns to exclude from normalization
    exclude_columns = ['Year_Birth', 'Kidhome', 'Teenhome', 'Dt_Customer', 'Response', 'AcceptedCmp3', 'AcceptedCmp4',	'AcceptedCmp5',	'AcceptedCmp1',	'AcceptedCmp2']

    exclude_columns.extend([col for col in processed_data.columns if any(x in col for x in ['Education', 'Kidhome', 'MStat', 'Teenhome'])])

    # Apply normalization to each column except the excluded ones
    for col in processed_data.columns:
        if col not in exclude_columns:
            processed_data[col] = normalizeData(processed_data[col])

    #Normalize the columns
    processed_data['Year_Birth'] = normalizeData(processed_data['Year_Birth'], 1)

    # converts dates to Unix timestamps
    timestamp = convert_to_timestamp(processed_data['Dt_Customer'])
    processed_data['Dt_Customer'] = normalizeData(timestamp, 1)

    print("Done normalizing the data...\n")

    processed_data = processed_data.fillna(0)

    ########## SAVE PROCESSED DATA IN A CSV FILE ###########
    processed_data.to_csv('processed_data.csv', index=False, header=True, sep=',', encoding='utf-8')

    # print("------------ Processed TEST --------------\n", processed_data[:20])


    #################       SPLIT DATA      #################
    print("Now Splitting data into Training, Validating, and Testing Samples...")
    # Step 1: Split the data into training and temp sets (train 70%, temp 30%)
    train_data, temp_data = train_test_split(processed_data, test_size=0.3, random_state=42)

    # Step 2: Split the temp set into validation and test sets (15% each)
    validation_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    # print("Number of training samples is", len(train_data))
    # print("Number of testing samples is", len(test_data))
    # print("Number of validating samples is", len(validation_data))
    # print("------------ TRAIN DATA --------------\n", train_data[:10])
    # print("\n------------ VALIDATING DATA --------------\n", validation_data[:10])
    # print("\n------------ TEST DATA --------------\n", test_data[:10])

    return train_data, validation_data, test_data



################### NN FUNCTIONS ###########################
# Activation function (Sigmoid) and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-np.array(x, dtype=float)))

def sigmoid_derivative(x):
    return x * (1 - x)

# Mean Squared Error Loss
def mse_loss(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))

# Accuracy function
def accuracy(y_true, y_pred):
    y_pred_rounded = np.round(y_pred)
    return np.mean(y_true == y_pred_rounded) * 100

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
