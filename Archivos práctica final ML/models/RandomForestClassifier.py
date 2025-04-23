# models/RandomForestClassifier.py

from sklearn.ensemble import RandomForestClassifier
import pickle


class RandomForest_Classifier:
    def __init__(self, n_estimators=100, max_depth=None):
        self.model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)

    def fit(self, X, y):
        self.model.fit(X,y)

    def predict(self, X):
        return self.model.predict(X)


    def save(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self.model, f)

    def load(self, file_path):
        with open(file_path, 'rb') as f:
            self.model = pickle.load(f)
