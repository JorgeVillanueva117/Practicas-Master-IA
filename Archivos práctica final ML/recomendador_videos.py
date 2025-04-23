import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


class RecomendadorVideos:
    def __init__(self, model_path, scaler_path, data_path):
        #cargamos el data set
        self.data_path = data_path
        self.df = pd.read_csv(self.data_path)

        #nos aseguramos que video_id este en formato numerico
        self.df['video_id'] = self.df['video_id'].astype('category').cat.codes

        #nos aseguramos de que 'comment_count' esté presente si es necesario
        if 'comment_count' not in self.df.columns:
            self.df['comment_count'] = np.ramdon.randint(0, 1000, size=len(self.df)) # asignasr valores aleatorios en el data set si no existe.

        # cargamos el modelo y el scaler
        self.model = self.load_pickle(model_path)
        self.scaler = self.load_pickle(scaler_path)

        # extraemos las caracteristicas
        self.features = ['views', 'likes', 'dislikes', 'comment_count']
        self.X = self.df[self.features]

        # Normalizar las características
        self.X_scaled = self.scaler.transform(self.X)

    def load_pickle(self, path):
        #cargar el modelo o escaler desde un archivo pickle
        with open(path, 'rb') as file:
            return pickle.load(file)

    def recomendar_videos(self, video_id):
        #recomendaciones basadas en un video_id numerico dado
        try:
            # convertir video_id a su codigo numérico
            video_id_code = self.df[self.df['video_id'] == video_id].index[0]

            # extraer caracteristicas del video y normalizarlas
            video_features = self.X.iloc[video_id_code].values.reshape(1, -1)

            # mantener la caracteristicas con los mismos nombres de cuando se entrenó el modelo
            video_scaled =self.scaler.transform(video_features)

            #usar KNN para encontrar los vecinos mas cercanos
            distances, indices = self.model.kneighbors(video_scaled, n_neighbors=6) # 6 vecinos mas cercanos incluyendo el video original

            recomendaciones = self.df.iloc[indices[0][1:]] # excluir el video original de la recomendación
            return recomendaciones [['video_id', 'title']].reset_index(drop=True)

        except IndexError:
            print(f"El video_id {video_id} no se encuentra en el data set.")
            return None


def main():
    # rutas de los archivos pickle modelo y scaler
    model_path = 'recomendador_videos_pkl/knn_model.pkl'
    scaler_path = 'recomendador_videos_pkl/scaler.pkl'
    data_path = 'data/archivo_procesado.csv'

    # instaciamos el recomendador
    recomendador = RecomendadorVideos(model_path, scaler_path, data_path)

    # Solicitamos el video_id al usuario
    video_id = input("introduce el video_id(numero) para obtener recomendaciones: ")

    # validadmos que el video_id es numérico
    try:
        video_id = int(video_id) # asegurarnos de que el video_id es un numero
        recomendaciones = recomendador.recomendar_videos(video_id)
        if recomendaciones is not None:
            print(f"Recomendaciones para el Video ID '{video_id}':")
            print(recomendaciones)
        else:
            print("No se encontraron recomendaciones")
    except ValueError:
        print("por favor, introduce un video_id válido (número).")


if __name__ == "__main__":
    main()
