from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
import os

# cargamos el data set
data_path = 'data/archivo_procesado.csv'
df = pd.read_csv(data_path)

# nos aseguramos que 'video_id este convertido a numeros
df['videos_id'] = df['videos_id'].astype('category').cat.codes

# nos aseguramos si comment_count esta si es necesario
if 'comment_count' not in df.columns:
    df['comment_count'] = np.random.randint(0, 1000, size=len(df))

#extraemos las caracteristicas
features = ['views','likes', 'dislikes', 'comment_count']
X = df[features]
y = df['category']  # ajustamos a la columna de la categoria correcta

#Normalizamos las caracteristicas
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#entrenamos el modelo KNN
model = KNeighborsClassifier(n_neighbors=6)
model.fit(X_scaled, y)

#creamos directorio si no existe
output_dir ='recomendador_videos_pkl'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

#guardamos el modelo y el escaler en el directorio especificado
model_path = os.path.join(output_dir, 'knn_model.pkl')
scaler_path = os.path.join(output_dir, 'scaler.pkl')

with open(model_path, 'wb') as model_file:
    pickle.dump(model, model_file)

with open(scaler_path, 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print(f"modelo y scaler guardados correctamente en: {output_dir}.")
