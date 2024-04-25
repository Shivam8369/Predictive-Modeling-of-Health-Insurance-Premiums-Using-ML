from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


app = Flask(__name__)

# Load your dataset
insurance_dataset = None  # Define insurance_dataset globally

# Function to preprocess data and train model
def preprocess_and_train_model():
    global insurance_dataset  # Access the global insurance_dataset variable

    # Load the dataset
    insurance_dataset = pd.read_csv('./cleaned_data.csv')

    # Encode categorical columns
    insurance_dataset.replace({'sex': {'male': 0, 'female': 1}}, inplace=True)
    insurance_dataset.replace({'smoker': {'yes': 0, 'no': 1}}, inplace=True)
    insurance_dataset.replace({'region': {'southeast': 0, 'southwest': 1, 'northeast': 2, 'northwest': 3}}, inplace=True)


    # Split data into features (X) and target (Y)
    X = insurance_dataset.drop(columns='charges', axis=1)
    Y = insurance_dataset['charges']

    # Split data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

    # Initialize and train your model
    regressor = LinearRegression()
    regressor.fit(X_train, Y_train)

    return regressor, X_train, Y_train

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Preprocess data and train model
        regressor, X_train, Y_train = preprocess_and_train_model()

        # Get user input from form
        age = float(request.form['age'])
        sex = int(request.form['sex'])
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        smoker = int(request.form['smoker'])
        region = int(request.form['region'])  # Note: 'region' is a string here


        # Create input data array
        input_data = np.array([age, sex, bmi, children, smoker,region]).reshape(1, -1)

        # Make prediction
        prediction = regressor.predict(input_data)
        predicted_charge = "{:.2f}".format(prediction[0])

        return render_template('index.html', prediction_text=f'The Annual Premium insurance will be approx ${predicted_charge}')

if __name__ == '__main__':
    app.run(debug=True)
