import configparser
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import os

# cargamos la configuracion desde conf_regresor.ini
config = configparser.ConfigParser

# ruta completa archivo de configuración
config_path = os.path.join(os.path.dirname(__file__), 'config', 'conf_regresor.ini')

# leemos el archivo de configuracion si no se encuentra lanza exception
if not os.path.exists(config_path):
    raise FileNotFoundError(f"El archivo de configuración no se encuentra en: {config_path}")

config.read(config_path)

# parámetros del archivo de comnfiguración
try:
    data_path = config['DEFAULT']['data_path']
    output_model_path = config['DEFAULT']['output_model_path']
    target_variable = config['DEFAULT']['target_variable']
except KeyError as e:
    raise KeyError(f"La clave {e} no se encuentra en el archivo de configuración")

# parametros de modelo KNN
n_neighbors = int(config['MODEL']['n_neighbors'])
weights = config['MODEL']['weights']
metric = config['MODEL']['metric']

# parametros de entrenamiento
test_size = float(config['TRINING']['test_size'])
random_state = int(config['TRAINING']['random_state'])
cv_folds = int(config['TRAINING']['cv_folds'])

# cargamos el data set
print(f"Cargndo datos desde: {data_path}")
data = pd.read_csv(data_path)

# Seleccinamos caracteristicas y vatiable objetivo
X = data[['views', 'comment_count']]  # caracteristicas
y = data[target_variable]  # variable a predecir 'likes

# escalamos las caracteristicas
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# dividimos los datos en conkjunto de entrenamiento y validación
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)

# creamos el modelo KNN Regressor
knn_regressor = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, metric=metric)

# entrenamos el modelo
print("Entrenando el model KNN regressor...")
knn_regressor.fit(X_train, y_train)

# Hacemos las predicciones
y_pred = knn_regressor.predict(X_test)

# evaluamos el modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"KNN Regressor - MSE: {mse:.4f}, R2: {r2:.4f}")

##Validacion cruzada
cv_scores = cross_val_score(knn_regressor, X_scaled, y, cv=cv_folds, scoring='neg_mean_squared_error')
cv_mse = np.mean(np.abs(cv_scores))
print(f"CV MSE (KNN Regressor): {cv_mse:.4f}")

# creamos la carpeta weights_regressor_likes si no existe
weights_dir = os.path.join(os.path.dirname(__file__), 'weights_regressor_likes')
if not os.path.exists(weights_dir):
    os.makedirs(weights_dir)

# guardamos el modelo entrenado y el scaler en 'weights_regressor_likes'
model_filname = 'knn_regressor_model.plk'
scaler_filaname = 'scaler.pkl'

# rutas completas para guardar archivos
output_model_path = os.path.join(weights_dir, model_filname)
output_scaler_path = os.path.join(weights_dir, scaler_filaname)

# guardamos el modelo
with open(output_model_path, 'wb') as f:
    pickle.dump(knn_regressor, f)

# guardamos el scaler
with open(output_scaler_path, 'wb') as f:
    pickle.dump(scaler, f)

print(f"Modelo guardado en: {output_model_path}")
print(f"Scaler guardado en: {output_scaler_path}")
