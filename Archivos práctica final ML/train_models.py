import sys
import os
import argparse
import configparser
from models.KNeighborsClassifier import KNN
from models.RandomForestClassifier import RandomForest_Classifier
from preprocess_data import preprocess_data

    # añadimos el directorio raiz al sys.path para evitar problemas de ruta con el paquete 'models
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    # argumentos para la xconfiguracion
    parser = argparse.ArgumentParser(description="Entrenamiento de modelos")
    parser.add_argument('--config', type=str, help='Ruta al archivo de configuración', required=True)
    args = parser.parse_args()

    # leer archivo de configuración
    config = configparser.ConfigParser()
    config.read(args.config)

    # obtenemos las rutas desde el archivo de configuración
    input_data_path = config['paths']['input_data'] # debe ser 'data/archivo_procesado.csv'
    knn_model_path = config['paths']['knn_model'] # ruta para guardar el modelo knn
    rf_model_path = config['paths']['rf_model'] # ruta para guardadr elmodelo randomforest

    # Preprocesar los datos
    preprocessed_data_path = 'data/archivo_procesado.csv' # Archivo de datos preprocesados
    preprocess_data(input_data_path, preprocessed_data_path)

    # cargamso los datos preprocesados
    import pandas as pd
    data = pd.read_csv(preprocessed_data_path)
    X = data[['views', 'likes', 'dislikes']] # caracteristicas de entrada
    y = data['category']# variable obkjetivo

    # entrenamos el modelo KNN
    knn_model = KNN(n_neighbors=5)
    knn_model.fit(X, y)
    knn_model.save(knn_model_path)
    print(f"Modelo KNN guardado en: {knn_model_path}")

    # entrenamos el modelo Random Forest
    rf_model = RandomForest_Classifier(n_estimators=100)
    rf_model.fit(X, y)
    rf_model.save(rf_model_path)
    print(f"Modelo Random forest guardado en: {rf_model_path}")

    print("Entrenamiento y guardado de modelos completado")

if __name__ == "__main__":
    main()
    # python train_models.py --config config/train.conf ,codigo de ejecución en consola