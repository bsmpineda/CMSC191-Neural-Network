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

    def train(self, X_train, y_train, X_val, y_val, epochs=1000, patience=5):
        best_val_loss = np.inf
        patience_counter = 0
        best_weights = None

        val_accuracy_history = []  # To store validation accuracy for each epoch
        val_loss_history = []      # To store validation loss for each epoch

        for epoch in range(epochs):
            # Forward pass for training data
            output = self.feedforward(X_train)
            self.backpropagate(X_train, y_train, output)

            # Calculate training and validation loss
            train_loss = mse_loss(y_train, output)
            val_output = self.feedforward(X_val)
            val_loss = mse_loss(y_val, val_output)
            val_acc = accuracy(y_val, val_output)

            # Store validation accuracy and loss
            val_accuracy_history.append(val_acc)
            val_loss_history.append(val_loss)
           

            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%')

            # Early stopping logic
            if round(val_loss, 5) < round(best_val_loss, 5):
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

        # Call the plotting functions for accuracy and loss
        plot_accuracy(val_accuracy_history)
        plot_loss(val_loss_history)
        # plot_accuracy(val_accuracy_history, start_epoch=40)
        # plot_loss(val_loss_history, start_epoch=40)

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

# Function to plot and save loss
def plot_loss(train_losses, val_losses, filename):
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def plot_accuracy(val_accuracy_history, start_epoch=1):
    """
    Function to plot and save validation accuracy over epochs.
    Args:
    - val_accuracy_history: list of validation accuracy per epoch
    - start_epoch: the epoch to start the plot from (for zooming)
    """
    epochs = range(start_epoch, len(val_accuracy_history) + 1)
    
    plt.figure(figsize=(6, 6))
    plt.plot(epochs, val_accuracy_history[start_epoch - 1:], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'Validation Accuracy over Epochs (from epoch {start_epoch})')
    plt.legend()
    plt.grid(True)
    
    # Save the accuracy plot
    plt.savefig(f'Results/validation_accuracy_plot_from_epoch_{start_epoch}.png')  # Save as a PNG file
    plt.show()

def plot_loss(val_loss_history, start_epoch=1):
    """
    Function to plot and save validation loss over epochs.
    Args:
    - val_loss_history: list of validation loss per epoch
    - start_epoch: the epoch to start the plot from (for zooming)
    """
    epochs = range(start_epoch, len(val_loss_history) + 1)
    
    plt.figure(figsize=(6, 6))
    plt.plot(epochs, val_loss_history[start_epoch - 1:], label='Validation Loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Validation Loss over Epochs (from epoch {start_epoch})')
    plt.legend()
    plt.grid(True)
    
    # Save the loss plot
    plt.savefig(f'Results/validation_loss_plot_from_epoch_{start_epoch}.png')  # Save as a PNG file
    plt.show()
