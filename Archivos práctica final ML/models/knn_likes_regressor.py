import pickle
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.model_selection import cross_val_score
import os  # necesario para crear la carpeta si no existe


class KNNRegressorModel:
    def __init__(self, param_grid=None):
        """
        inicializa el modelo knn con los parametros a considerar.
        si no lo hace con predefinidos
"""
        self.param_grid = param_grid if param_grid else {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'metrics': ['euclidean', 'manhattan', 'chebyshev']
        }
        self.model = None
        self.scaler = StandardScaler()

    def fit(self, X, y):
        """
        entrenamos el modelo con los dato proporcionados.
        recibe un data frame o numpy array de entradas X y un objetivo y.
        """
        # Escalamos las caracteristicas
        X_scaled = self.scaler.fit_transform(X)

        # Optimizacion del knnregresor con gridsearchcv
        grid_search = GridSearchCV(KNeighborsRegressor(), self.param_grid, cv=5, scoring='neg_mean_squared_error',
                                   n_jobs=-1)
        grid_search.fit(X_scaled, y)

        # Guarda el mejor modelo encontrado
        self.model = grid_search.best_estimator_

        # imprime los mejores parametros encontrados
        print(f" Mejores parametros encontrados:{grid_search.best_params_}")

    def predict(self, X):
        """
        Realiza predicciones utilizando el modelo entrenado.
        Puede realizar predicciones para un solo dato o un conjunto de datos.
        """
        if self.model is None:
            raise Exception("El modelo no ha sido entrenado")

        X_scaled = self.scaler.transform(X)# Escalamos los datos antes de predecir
        return self.model.predict(X_scaled)

    def save(self, model_filepath, scaler_filepath):
        """
        'guerdamos el modelo entrenado y el scaler en archivos pickle dentro de la carpeta weights_regressor_likes'
        """
        # nos aseguramos de que la carpeta 'weights_regressor_likes exista.
        os.makedirs('weights_regressor_likes', exist_ok=True)

        # Guardamos el modelo
        with open(model_filepath, 'wb') as f:
            pickle.dump(self.model, f)

        # guardamos el scaler
        with open(scaler_filepath, 'wb') as f:
            pickle.dump(self.scaler, f)

        print(f"Modelo guardado en: {model_filepath}")
        print(f"Scaler guardador en: {scaler_filepath}")

    def load(self, model_filepath, scaler_filepath):
        """
        Cargamos el modelo entrenado y el scaler desde los archivos pickle.
        """
        with open(model_filepath, 'rb') as f:
            self.model = pickle.load(f)

        with open(scaler_filepath, 'rb') as f:
            self.scaler = pickle.load(f)

    def evaluate(self, X_test, y_test):
        """
         Evsluamos el modelo utilizando las metricas MSE y R2.
        """
        y_pred = self.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return mse, r2

    def cross_validate(self, X, y, cv=5):
        """
        REaliza la validación cruzada y devuelve la media MSE
        """
        X_scaled = self.scaler.transform(X)  # escalar caracteristicas
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=cv, scoring='neg_mean_squared_error')
        cv_mse = np.mean(np.abs(cv_scores))
        return cv_mse
