import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def preprocess_data(csv_file_path, output_file_path):
    # verificamos si exsiste la carpeta si no se crea
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    # cargamos los datos
    df = pd.read_csv(csv_file_path)

    # trnsformaciones comunes como la normalización
    scaler = StandardScaler()
    df[['views', 'likes', 'dislikes']] = scaler.fit_transform(df[['views', 'likes', 'dislikes']])

    # guardamso el dataset procesado
    df.to_csv(output_file_path, index=False)
    print(f"Datos preprocesados guardados en: {output_file_path}")


#rutas de entrada y salida
csv_file_path = 'data/total_videos_unificado.csv'  # ruta entrada archivo csv
output_file_path = 'data/archivo_procesado.csv'  # ruta del salida del archivo

# llamar a la funcion preprocess_data
preprocess_data(csv_file_path, output_file_path)
