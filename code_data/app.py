from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter

app = Flask(__name__)

# Hàm xử lý tính toán thông tin gain
def split_node(column, threshold_split):
    left_node = column[column <= threshold_split].index
    right_node = column[column > threshold_split].index
    return left_node, right_node

def entropy(y_target):
    values, counts = np.unique(y_target, return_counts=True)
    return -np.sum([(count / len(y_target)) * np.log2(count / len(y_target)) for count in counts])

def info_gain(column, target, threshold_split):
    entropy_start = entropy(target)
    left_node, right_node = split_node(column, threshold_split)
    
    # Tính độ dài và trọng số entropy
    n_target = len(target)
    n_left = len(left_node)
    n_right = len(right_node)
    
    entropy_left = entropy(target[left_node])
    entropy_right = entropy(target[right_node])
    
    weight_entropy = (n_left / n_target) * entropy_left + (n_right / n_target) * entropy_right
    
    return entropy_start - weight_entropy

def best_split(X, target, feature_ids):
    best_ig = -1
    best_feature = None
    best_threshold = None
    for feat_id in feature_ids:
        column = X.iloc[:, feat_id]
        for threshold in set(column):
            ig = info_gain(column, target, threshold)
            if ig > best_ig:
                best_ig = ig
                best_feature = X.columns[feat_id]
                best_threshold = threshold
    return best_feature, best_threshold

def most_value(y_target):
    return y_target.value_counts().idxmax()

# Class node cây quyết định
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    
    def is_leaf_node(self):
        return self.value is not None

# Class Decision Tree
class DecisionTreeClass:
    def __init__(self, min_samples_split=2, max_depth=10, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None
    
    def grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_classes = len(np.unique(y))
        
        if depth >= self.max_depth or n_classes == 1 or n_samples < self.min_samples_split:
            leaf_value = most_value(y)
            return Node(value=leaf_value)
        
        feature_ids = np.random.choice(n_feats, self.n_features, replace=False)
        best_feature, best_threshold = best_split(X, y, feature_ids)
        
        left_node = X[best_feature] <= best_threshold
        right_node = X[best_feature] > best_threshold
        
        left = self.grow_tree(X.loc[left_node], y.loc[left_node], depth + 1)
        right = self.grow_tree(X.loc[right_node], y.loc[right_node], depth + 1)
        
        return Node(best_feature, best_threshold, left, right)
    
    def fit(self, X, y):
        self.n_features = X.shape[1] if self.n_features is None else min(X.shape[1], self.n_features)
        self.root = self.grow_tree(X, y)
    
    def predict(self, X):
        return np.array([self.traverse_tree(row, self.root) for _, row in X.iterrows()])
    
    def traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self.traverse_tree(x, node.left)
        return self.traverse_tree(x, node.right)

# Class Random Forest
class RandomForest:
    def __init__(self, n_trees=5, max_depth=10, min_samples_split=2, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []
    
    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTreeClass(min_samples_split=self.min_samples_split, max_depth=self.max_depth, n_features=self.n_features)
            X_sample, y_sample = bootstrap(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
    
    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.array([most_value(pd.Series(tree_preds[:, i])) for i in range(X.shape[0])])

def bootstrap(X, y):
    n_samples = X.shape[0]
    idxs = np.random.choice(n_samples, n_samples, replace=True)
    return X.iloc[idxs], y.iloc[idxs]

# Hàm train mô hình và dự đoán
def train_and_predict():
    # Load dữ liệu
    data = pd.read_csv('drug200.csv')
    X = data.drop(columns=['Drug'])
    y = data['Drug']
    
    # Biến đổi dữ liệu
    X['Sex'] = X['Sex'].replace({'M': 0, 'F': 1})
    X['BP'] = X['BP'].replace({'HIGH': 2, 'NORMAL': 1, 'LOW': 0})
    X['Cholesterol'] = X['Cholesterol'].replace({'HIGH': 1, 'NORMAL': 0})
    y = y.replace({'drugA': 0, 'drugB': 1, 'drugC': 2, 'drugX': 3, 'DrugY': 4})
    
    # Chia tập train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Huấn luyện mô hình RandomForest
    rf = RandomForest(n_trees=10)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    # Tính độ chính xác
    accuracy_rf = np.mean(y_test.values == y_pred) * 100
    return y_pred, accuracy_rf

# Route Flask
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    y_pred, acc = train_and_predict()
    return render_template('result.html', prediction=y_pred, accuracy=acc)

if __name__ == "__main__":
    app.run(debug=True)
