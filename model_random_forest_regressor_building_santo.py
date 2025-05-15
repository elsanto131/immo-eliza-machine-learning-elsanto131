from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import joblib

def build_random_forest_model(df):
    # Séparation des variables explicatives (X) et de la variable cible (y)
    X = df.drop(columns='price')
    y = df['price']

    # Division des données en ensemble d'entraînement et ensemble de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fonction d’imputation des valeurs manquantes
    def mean_and_mode(X_train, X_test):
        X_train = X_train.copy()
        X_test = X_test.copy()

        # Colonnes numériques critiques à imputer par la médiane
        important_numeric_cols = [
            'bedroomcount',
            'bathroomcount',
            'cadastralincome',
            'primaryenergyconsumptionpersqm',
        ]

        for col in important_numeric_cols:
            if col in X_train.columns:
                median = X_train[col].median()
                X_train[col] = X_train[col].fillna(median)
                X_test[col] = X_test[col].fillna(median)

        # Autres colonnes numériques
        other_numeric_cols = [col for col in X_train.select_dtypes(include=[np.number]).columns if col not in important_numeric_cols]
        for col in other_numeric_cols:
            median = X_train[col].median()
            X_train[col] = X_train[col].fillna(median)
            X_test[col] = X_test[col].fillna(median)

        # Colonnes catégorielles : imputation par la valeur la plus fréquente (mode)
        categorical_cols = X_train.select_dtypes(include=['object', 'category', 'bool']).columns
        for col in categorical_cols:
            mode = X_train[col].mode()[0]
            X_train[col] = X_train[col].fillna(mode)
            X_test[col] = X_test[col].fillna(mode)

        return X_train, X_test

    # Appel effectif de l’imputation
    X_train, X_test = mean_and_mode(X_train, X_test)

    # Encodage des colonnes catégorielles (important pour RandomForest)
    categorical_cols = X_train.select_dtypes(include=['object', 'category', 'bool']).columns
    for col in categorical_cols:
        X_train[col] = X_train[col].astype('category').cat.codes
        X_test[col] = X_test[col].astype('category').cat.codes

    # Définition du modèle
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        random_state=42
    )

    # Entraînement du modèle
    rf_model.fit(X_train, y_train)
    joblib.dump(X_train.columns.tolist(), 'model_features.pkl') # EN Save the feature names with Joblib - FR Enregistrer les noms des caractéristiques avec Joblib
    

    # Prédictions
    y_pred = rf_model.predict(X_test)
    y_pred_train = rf_model.predict(X_train)

    # Évaluation des performances
    y_r2 = r2_score(y_test, y_pred)
    X_r2 = r2_score(y_train, y_pred_train)
    y_mae = mean_absolute_error(y_test, y_pred)
    X_mae = mean_absolute_error(y_train, y_pred_train)
    y_mse = mean_squared_error(y_test, y_pred)
    X_mse = mean_squared_error(y_train, y_pred_train)
    y_rmse = np.sqrt(y_mse)
    X_rmse = np.sqrt(X_mse)

    # Affichage des résultats
    print(f"R2 Score : {y_r2}")
    print(f"R2 Score : {X_r2}")
    print(f"Mean Absolute Error : {y_mae}")
    print(f"Mean Absolute Error : {X_mae}")
    print(f"Mean Squared Error : {y_mse}")
    print(f"Mean Squared Error : {X_mse}")
    print(f"Root Mean Squared Error : {y_rmse}")
    print(f"Root Mean Squared Error : {X_rmse}")
    
    return 'Random Forest', rf_model, X_r2, y_r2, X_rmse, y_rmse, X_mae, y_mae
