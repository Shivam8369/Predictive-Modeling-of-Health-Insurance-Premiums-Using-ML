import pandas as pd
import numpy as np


# Read the CSV file into a DataFrame
df = pd.read_csv('./unclean_data.csv' , delimiter=',')  # specify your own


# Display the original DataFrame
print("Original DataFrame:")
print(df)


# Convert the 'AGE' column to numeric, replacing non-numeric values with NaN
df['age'] = pd.to_numeric(df['age'], errors='coerce')


# Task 1: Find the average age (excluding values greater than or equal to 100)
average_age = df.loc[df['age'] < 100, 'age'].mean()


# Display the average age
print("\nAverage Age (excluding values >= 100):", average_age)


# Replace values in the 'AGE' column with the calculated average
df.loc[df['age'] >= 100, 'age'] = average_age
df['age'] = df['age'].astype(int)




# Task 2: Find the most frequent gender
most_frequent_gender = df['sex'].mode()[0]


# Display the most frequent gender
print("\nMost Frequent Gender:", most_frequent_gender)


# Replace NaN values in the 'GENDER' column with the most frequent gender
df['sex'].fillna(most_frequent_gender, inplace=True)


# Display the DataFrame after cleaning
print("\nCleaned DataFrame:")
print(df)


# Update the CSV file with the cleaned DataFrame
df.to_csv('cleaned_data.csv', index=False)
