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
from sklearn.model_selection import train_test_split
from functions import *

# Neural Network hyperparameters
epochs = 1000
learnrate = 0.0001

# Reading the csv file into a pandas DataFrame
data = pd.read_csv('shopAdvertisement_data.csv', sep='\t')
data.head()

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

########## Splitting the Train data into Features (Inputs) and Targets (Outputs) ##########

# List of target columns 
target_columns = ['Response', 'AcceptedCmp3', 'AcceptedCmp4',	'AcceptedCmp5',	'AcceptedCmp1',	'AcceptedCmp2']

# Splitting features and targets for training data
features = train_data.drop(target_columns, axis=1)  # Drop the target columns to get the features/inputs
targets = train_data[target_columns]  # Select the target columns/outputs

# Splitting features and targets for test data
features_test = test_data.drop(target_columns, axis=1)  # Drop the target columns to get the features
targets_test = test_data[target_columns]  # Select the target columns

print("------------ FEATURES TEST --------------\n", features_test[:10])
print("------------ TARGET TEST --------------\n", targets_test['Response'][:10])

print("\n--------------- START TRAINING -----------------------\n")

weights = train_nn(features, targets['Response'], epochs, learnrate)

print("--------------- DONE TRAINING -----------------------\n")

# Calculate accuracy on test data
test_out = sigmoid((np.dot(features_test, weights)).astype(float))
predictions = test_out > 0.5
accuracy = np.mean(predictions == targets_test['Response'])
print("Prediction accuracy: {:.3f}".format(accuracy))