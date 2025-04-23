import argparse
import pickle
import pandas as pd
import os  # importamos os para manejar rutas y crear directorios


# funcion para cargar un modelo desde un archivo pickle
def load_model(model_path):
    #carga un modelo guardado desde un archivo pickle
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model


# funcion para procesar los datos de entrada
def preprocess_input_data(input_data):
    #preprocesa los datos de entrada para la inferencia

    # seleccionamos las carateristicas:
    processed_data = input_data[['views','likes', 'dislikes']]
    return processed_data


# funcion para realizar la inferencia con un modelo
def make_prediction(model, data):
    #hace la prediccion usando el modelo cargado y los datos preprocesados."""
    predictions = model.predict(data)
    return predictions


# función para manejar la inferencia
def run_inference(model_type, model_path, input_csv):
    """realiza la inferencia con el modelo seleccionado y los datos de entrada ."""
    # carga los nuevos datos para predecir
    new_data = pd.read_csv(input_csv)

    # Preprocesamos los datos
    processed_data = preprocess_input_data(new_data)

    # cargamos el modelo guardado
    model = load_model(model_path)

    # hacemos las prediciones con el modelo
    predictions = make_prediction(model, processed_data)

    # Mostramso las predicciones
    print(f"Predicciones del modelo {model_type}:")
    print(predictions)

    # nos aseguramos que la carpeta Predicciones_models existe
    os.makedirs('Predicciones_models', exist_ok=True)

    # guardamos las predicciones en la carpeta Predicciones_models
    output = pd.DataFrame({'predictions': predictions})
    output_file_path = f'Predicciones_models/predicciones_{model_type}.csv'
    output.to_csv(output_file_path, index=False)
    print(f"Predicciones guardadas en '{output_file_path}'.")


#configuramos argparse para recibir los argumentos
def parse_args():
    #configuramos los argumentos para la linea de comandos
    parser = argparse.ArgumentParser(description="Realiza inferencia con un modelo previamente entrenado.")

    # Argumentos
    parser.add_argument('--model_type', type=str, required=True, choices=['knn', 'rf'],
                        help="Tipo de modelo a utilizar ('knn', 'rf').")
    parser.add_argument('--model_path', type=str, required=True,
                        help="Ruta del archivo con los pesos del modelo (.pkl).")
    parser.add_argument('--input_csv', type=str, required=True, help="ruta del archivo csv con los datos de entrada.")

    return parser.parse_args()


# punto de entrada priincipal
if __name__ == "__main__":
    # procesamos los argumentos
    args = parse_args()

    # ejecutamos la inferenciia para el modelo solicitado
    run_inference(args.model_type, args.model_path, args.input_csv)

## python inference_model.py --model_type knn --model_path weights/knn_model.pkl --input_csv data/archivo_procesado.csv codigo ejecucion en consola

## python inference_model.py --model_type rf --model_path weights/rf_model.pkl --input_csv data/archivo_procesado.csv codigo ejecucion en consola

