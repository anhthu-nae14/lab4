# models.py

class DecisionTreeClass:
    def __init__(self, min_samples_split=2, max_depth=10):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        # Logic của hàm fit để huấn luyện Decision Tree
        pass

    def predict(self, X):
        # Logic của hàm predict để dự đoán
        pass

class RandomForest:
    def __init__(self, n_trees=10, n_features=None):
        self.n_trees = n_trees
        self.n_features = n_features
        self.trees = []

    def fit(self, X, y):
        # Logic của hàm fit để huấn luyện Random Forest
        pass

    def predict(self, X):
        # Logic của hàm predict để dự đoán
        pass
