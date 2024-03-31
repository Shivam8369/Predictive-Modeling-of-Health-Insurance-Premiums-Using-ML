# step1 => Importing the Dependencies/libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics



# step2 => Data Collection and Manipulation

# loading the data from csv file to a Pandas DataFrame
insurance_dataset = pd.read_csv('./insurance.csv')

# first 5 rows of the dataframe
insurance_dataset.head()

# number of rows and columns
insurance_dataset.shape

# getting some information about the dataset
insurance_dataset.info()

# checking for missing values
insurance_dataset.isnull().sum()

insurance_dataset.duplicated().sum()

insurance_dataset.drop_duplicates()

insurance_dataset.isnull().sum()



# step3 => Data Analysis/visualization Part

# statistical Measures of the dataset
insurance_dataset.describe()

sns.set_theme()
plt.figure(figsize=(6,6))
sns.histplot(x='age', data=insurance_dataset)
plt.title('Age Distribution')
plt.show()

# Gender column
plt.figure(figsize=(5,5))
sns.countplot(x='sex', data=insurance_dataset)
plt.title('Sex Distribution')
plt.show()

# bmi distribution
plt.figure(figsize=(5,5))
sns.histplot(insurance_dataset['bmi'])
plt.title('BMI Distribution')
plt.show()

# children column
plt.figure(figsize=(6,6))
sns.countplot(x='children', data=insurance_dataset)
plt.title('Children')
plt.show()

# smoker column
plt.figure(figsize=(5,5))
sns.countplot(x='smoker', data=insurance_dataset)
plt.title('Smoker Distribution')
plt.show()

# region column
plt.figure(figsize=(5,5))
sns.countplot(x='region', data=insurance_dataset)
plt.title('Region Distribution')
plt.show()

# distribution of charges value
plt.figure(figsize=(6,6))
sns.histplot(insurance_dataset['charges'])
plt.title('Charges Distribution')
plt.show()



# step4 => Data Pre-Processing

 
# Encoding the categorical features to Numeric value

# encoding sex column
insurance_dataset.replace({'sex':{'male':0,'female':1}}, inplace=True)

 # encoding 'smoker' column
insurance_dataset.replace({'smoker':{'yes':0,'no':1}}, inplace=True)

# encoding 'region' column
insurance_dataset.replace({'region':{'southeast':0,'southwest':1,'northeast':2,'northwest':3}}, inplace=True)

# checking the dataset
insurance_dataset.tail()

# Splitting the Features and Target

X = insurance_dataset.drop(columns='charges', axis=1)
Y = insurance_dataset['charges']
print("Features")
print(X)
print()
print("Labels")
print(Y)


#  step5 =>  Splitting the data into Training data & Testing Data


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

print(X.shape, X_train.shape, X_test.shape)


#  step6 =>  Model Training

## Linear Regression Algorithm

# loading the Linear Regression model
regressor = LinearRegression()

regressor.fit(X_train, Y_train)


#  step7 => Model Evaluation

# prediction on training data
training_data_prediction =regressor.predict(X_train)

# R squared value
r2_train = metrics.r2_score(Y_train, training_data_prediction)
print('R squared value : ', r2_train)

# prediction on test data
test_data_prediction =regressor.predict(X_test)

# R squared value
r2_test = metrics.r2_score(Y_test, test_data_prediction)
print('R squared value : ', r2_test)




#  step8 => Testing The Predictive System

# Take input from the user
age = float(input("Enter age: "))
sex = int(input("Enter sex (0 for male, 1 for female): "))
bmi = float(input("Enter BMI: "))
children = int(input("Enter number of children: "))
smoker = int(input("Enter smoker status (0 for yes, 1 for no): "))
region = int(input("Enter region (0 for southeast, 1 for southwest, 2 for northeast, 3 for northwest): "))

# Create a tuple with user input
input_data = (age, sex, bmi, children, smoker, region)

# Convert input data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape the array
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Make prediction
prediction = regressor.predict(input_data_reshaped)

print('The Annual Premium insurance will be approx $', "{:.2f}".format(prediction[0]))
