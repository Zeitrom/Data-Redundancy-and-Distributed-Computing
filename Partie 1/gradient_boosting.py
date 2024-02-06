# -*- coding: utf-8 -*-
"""gradient_boosting.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1dyWXMg-E4zIXky14Ke0bZ8rg_fnZgx1L
"""

# Importation des bibliothèques nécessaires
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

#Flask
from flask import Flask, request, jsonify

app = Flask(__name__)

# Chargement des données depuis un fichier CSV
data = pd.read_csv('Housing.csv', delimiter= ',')

# Nettoyage des données : suppression des lignes avec des valeurs manquantes et conversion des valeurs catégorielles en numériques
df = pd.DataFrame(data)
df = df.dropna()
df['mainroad'] = df['mainroad'].map({'yes': 1, 'no': 0})
df['guestroom'] = df['guestroom'].map({'yes': 1, 'no': 0})
df['basement'] = df['basement'].map({'yes': 1, 'no': 0})
df['hotwaterheating'] = df['hotwaterheating'].map({'yes': 1, 'no': 0})
df['airconditioning'] = df['airconditioning'].map({'yes': 1, 'no': 0})
df['prefarea'] = df['prefarea'].map({'yes': 1, 'no': 0})
df['furnishingstatus'] = df['furnishingstatus'].map({'furnished': 1, 'semi-furnished': 0})

# Remplissage des valeurs manquantes avec la moyenne pour certaines colonnes
df['area'].fillna(df['area'].mean(), inplace=True)
df['bedrooms'].fillna(df['bedrooms'].mean(), inplace=True)
df['bathrooms'].fillna(df['bathrooms'].mean(), inplace=True)
df['stories'].fillna(df['stories'].mean(), inplace=True)

# Remplissage des valeurs manquantes avec 0.5 pour certaines colonnes binaires
df['mainroad'].fillna(0.5, inplace=True)
df['guestroom'].fillna(0.5, inplace=True)
df['basement'].fillna(0.5, inplace=True)
df['hotwaterheating'].fillna(0.5, inplace=True)
df['airconditioning'].fillna(0.5, inplace=True)

# Suppression des lignes avec des valeurs manquantes restantes
df = df.dropna()

# Séparation des caractéristiques (X) et de la variable cible (y)
X = df.drop('price', axis=1)
y = df['price']

# Division de l'ensemble de données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraînement du modèle Gradient Boosting
gb_model = GradientBoostingRegressor()
gb_model.fit(X_train, y_train)

# Prédictions sur l'ensemble de test avec le modèle Gradient Boosting
dt_prediction = gb_model.predict(X_test)

# Impression du prix prédit pour la première instance dans l'ensemble de test
print("Prix prédit par le Boosting de Gradient:", dt_prediction[0])

# Calcul du coefficient de détermination R2 pour le modèle Gradient Boosting
dt_r2 = r2_score(y_test, dt_prediction)

# Affichage du coefficient de détermination R2
print("R2 pour le modèle de Boosting de Gradient:", dt_r2)

# Route pour faire des prédictions via une API REST
@app.route('/http://127.0.0.1:5000/predict', methods=['GET'])
def predict():
    try:
        # Extraction des paramètres de la requête
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

        # Prédiction avec le modèle entraîné
        input_data = np.array([[area, bedrooms, bathrooms, stories, mainroad, guestroom, basement,
                                hotwaterheating, airconditioning, parking, prefarea, furnishingstatus]])
        prediction = gb_model.predict(input_data)[0]

        # Réponse au format JSON
        response = {'prédiction': prediction}
        return jsonify(response)

    except Exception as e:
        # En cas d'erreur, renvoyer un message d'erreur au format JSON
        return jsonify({'erreur': str(e)})


if __name__ == '__main__':
    # Démarrage de l'application Flask en mode debug
    app.run(debug=True)

