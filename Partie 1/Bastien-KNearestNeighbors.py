from sklearn.neighbors import KNeighborsRegressor
# Flask
from flask import Flask, request, jsonify

app = Flask(__name__)

# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Assuming you have a dataset with the given variables
# Replace this with your actual dataset
data = pd.read_csv("Partie 1\Housing.csv", delimiter= ',')

df = pd.DataFrame(data)
df = df.dropna()
df['mainroad'] = df['mainroad'].map({'yes': 1, 'no': 0})
df['guestroom'] = df['guestroom'].map({'yes': 1, 'no': 0})
df['basement'] = df['basement'].map({'yes': 1, 'no': 0})
df['hotwaterheating'] = df['hotwaterheating'].map({'yes': 1, 'no': 0})
df['airconditioning'] = df['airconditioning'].map({'yes': 1, 'no': 0})
df['prefarea'] = df['prefarea'].map({'yes': 1, 'no': 0})
df['furnishingstatus'] = df['furnishingstatus'].map({'furnished': 1, 'semi-furnished': 0})

df['area'].fillna(df['area'].mean(), inplace=True)
df['bedrooms'].fillna(df['bedrooms'].mean(), inplace=True)
df['bathrooms'].fillna(df['bathrooms'].mean(), inplace=True)
df['stories'].fillna(df['stories'].mean(), inplace=True)

df['mainroad'].fillna(0.5, inplace=True)
df['guestroom'].fillna(0.5, inplace=True)
df['basement'].fillna(0.5, inplace=True)
df['hotwaterheating'].fillna(0.5, inplace=True)
df['airconditioning'].fillna(0.5, inplace=True)

# Drop rows with any remaining NaN values
df = df.dropna()

# Separate features (X) and target variable (y)
# Include all relevant features in X
X = df.drop('price', axis=1)
y = df['price']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creation and training of the KNN model
knn_regressor = KNeighborsRegressor(n_neighbors=5)
knn_regressor.fit(X_train, y_train)

# Make predictions using the trained KNN model on the test set
prediction = knn_regressor.predict(X_test)

# Calculate R2 for the KNN model
knn_r2 = r2_score(y_test, prediction)

# Display R2
print("R2 for KNN:", knn_r2)

@app.route('/predict', methods=['GET'])
def predict():
    try:
        area = float(request.args.get('area'))
        bedrooms = float(request.args.get('bedrooms'))
        bathrooms = float(request.args.get('bathrooms'))
        stories = float(request.args.get('stories'))
        mainroad = float(request.args.get('mainroad'))
        guestroom = float(request.args.get('guestroom'))
        basement = float(request.args.get('basement'))
        hotwaterheating = float(request.args.get('hotwaterheating'))
        airconditioning = float(request.args.get('airconditioning'))
        parking = float(request.args.get('parking'))
        prefarea = float(request.args.get('prefarea'))
        furnishingstatus = float(request.args.get('furnishingstatus'))

        # Perform the prediction
        input_data = np.array([[area, bedrooms, bathrooms, stories, mainroad, guestroom, basement,
                                hotwaterheating, airconditioning, parking, prefarea, furnishingstatus]])
        prediction = knn_regressor.predict(input_data)[0]

        # Reply
        response = {'prediction': prediction}
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
