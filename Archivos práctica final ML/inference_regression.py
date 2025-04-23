import configparser
import pandas as pd
import pickle
import os
from sklearn.preprocessing import StandardScaler

# cargamos la configuración desde el archivo conf_regresor.ini
config = configparser.ConfigParser()
config.read('config/conf_regresor.ini')

# prarametros del archivo de configuración
data_path = config['DEFAULT']['data_path']
output_model_path = config['DEFAULT']['output_model_path']
target_variable = config['DEFAULT']['target_variable']

# cargamos el data set
print(f"cargamos los datos desde:{data_path}")
data = pd.read_csv(data_path)

# seleccionamos las caracteristicas y la variable objetivo
X = data[['views', 'comment_count']] # dejamos solo  'views' 'comment_count' como caracteristicas
y = data[target_variable]  # la variable a predecir es 'likes''

# escalamos las caracteristicas
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# creamos la ruta del modelo desde la carpeta ' weights'
weights_dir = os.path.join(os.path.dirname(__file__), 'weights')
model_filename = 'knn_regressor_model.pkl'
model_path = os.path.join(weights_dir, model_filename)

# cargamos el modelo entrenado si no se encuentre lanza excepcion
if not os.path.exists(model_path):
    raise FileNotFoundError(f"El archivo de modelo no sen encuentra en: {model_path}")

with open(model_path, 'rb') as f:
    knn_regressor = pickle.load(f)

# Hacer predicciones
y_pred = knn_regressor.predict(X_scaled)

# cramos la carpeta si no existe.
output_dir = 'Prediccion_likes_KNN_regressor'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# guardamos las predicciones en la carpeta correspondiente
predictions_df =pd.DataFrame(y_pred, columns=[target_variable])
predictions_path = os.path.join(output_dir, 'predicciones_likes.csv')
predictions_df.to_csv(predictions_path, index=False)

print(f"Predicciones guardadas en: {predictions_path}")
