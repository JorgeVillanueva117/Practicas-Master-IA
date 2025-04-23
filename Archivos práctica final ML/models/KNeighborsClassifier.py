import pickle
from sklearn.neighbors import KNeighborsClassifier


class KNN:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.model = None

    def fit(self, X_train, y_train):
        """entrenamos el modelo con los datos proporcionsdos"""
        self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        self.model.fit(X_train, y_train)

    def predict(self, X):
        """ realizamos predicciones con el modelo ya entrenado."""
        if self.model is None:
            raise Exception("El modelo no ha sido entrenado")
        return self.model.predict(X)

    def save(self, filepath):
        """Guardamos el modelo entrenado en un archivo"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)

    def load(self, filepath):
        """cargamos un modelo previamente guardado."""
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
