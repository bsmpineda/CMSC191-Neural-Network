'''
PINEDA, BRIXTER SIEN M.
2020-2019
CMSC 191 - NEURAL NETWORS

A Neural Network Predicting the Result of a Shop's Advertisement

'''

# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functions import *

# Neural Network hyperparameters
epochs = 1000
learnrate = 0.0001

# Reading the csv file into a pandas DataFrame
data = pd.read_csv('shopAdvertisement_data.csv', sep='\t')
data.head()

# Drop ID, ID is not included
data = data.drop("ID", axis="columns") 

############# ONE-HOT ENCODING THE RANK ################
# Make dummy variables for Education
one_hot_data = pd.concat([data, pd.get_dummies(data['Education'], prefix='Education')], axis=1)

# Drop the previous Education column
one_hot_data = one_hot_data.drop("Education", axis='columns')

# Make dummy variables for Marital Status
one_hot_data = pd.concat([one_hot_data, pd.get_dummies(one_hot_data['Marital_Status'], prefix='MStat')], axis=1)

# Drop the previous Marital Status column
one_hot_data = one_hot_data.drop("Marital_Status", axis='columns')


###############  NORMALIZATION and SCALING DATA  ################
# THIS IS TO MAKE SURE THAT DATA ARE IN [0, 1]

# Making a copy of our data
processed_data = one_hot_data[:]

# Scale the columns

# Columns to exclude from normalization
exclude_columns = ['Year_Birth', 'Kidhome', 'Teenhome', 'Dt_Customer', 'Response', 'AcceptedCmp3', 'AcceptedCmp4',	'AcceptedCmp5',	'AcceptedCmp1,'	'AcceptedCmp2']

# Add columns containing "Education" to the exclusion list
exclude_columns.extend([col for col in processed_data.columns if 'Education' in col])

# Add columns containing "MStat" to the exclusion list
exclude_columns.extend([col for col in processed_data.columns if 'MStat' in col])

# Apply normalization to each column except the excluded ones
for col in processed_data.columns:
    if col not in exclude_columns:
        processed_data[col] = normalizeData(processed_data[col])

#Normalize the columns
processed_data['Year_Birth'] = normalizeData(processed_data['Year_Birth'], 1)

# converts dates to Unix timestamps
timestamp = convert_to_timestamp(processed_data['Dt_Customer'])
processed_data['Dt_Customer'] = normalizeData(timestamp, 1)

########### SAVE PROCESSED DATA IN A CSV FILE ###########
# processed_data.to_csv('processed_data.csv', index=False, header=True, sep=',', encoding='utf-8')