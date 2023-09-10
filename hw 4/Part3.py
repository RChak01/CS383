import numpy as np
import pandas as pd

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, Y):
        self.tree = self._build_tree(X, Y, depth=0)

    def _build_tree(self, X, Y, depth):
        if self.max_depth is not None and depth >= self.max_depth:
            return np.bincount(Y).argmax()

        unique_classes, class_counts = np.unique(Y, return_counts=True)

        if len(unique_classes) == 1:
            return unique_classes[0]

        best_split = None
        best_gini = 1.0

        for feature in range(X.shape[1]):
            for threshold in np.unique(X[:, feature]):
                left_indices = X[:, feature] < threshold
                left_gini = self._gini_impurity(Y[left_indices])
                right_gini = self._gini_impurity(Y[~left_indices])
                gini = (left_gini * sum(left_indices) + right_gini * sum(~left_indices)) / len(Y)
                if gini < best_gini:
                    best_split = (feature, threshold)
                    best_gini = gini

        if best_split is None:
            return np.bincount(Y).argmax()

        feature, threshold = best_split
        left_indices = X[:, feature] < threshold
        right_indices = ~left_indices

        left_tree = self._build_tree(X[left_indices], Y[left_indices], depth + 1)
        right_tree = self._build_tree(X[right_indices], Y[right_indices], depth + 1)

        return (feature, threshold, left_tree, right_tree)

    def _gini_impurity(self, Y):
        _, counts = np.unique(Y, return_counts=True)
        probabilities = counts / len(Y)
        return 1 - np.sum(probabilities ** 2)

    def predict(self, X):
        predictions = [self._predict_tree(x, self.tree) for x in X]
        return np.array(predictions)

    def _predict_tree(self, x, tree):
        if not isinstance(tree, tuple):
            return tree
        feature, threshold, left_tree, right_tree = tree
        if x[feature] < threshold:
            return self._predict_tree(x, left_tree)
        else:
            return self._predict_tree(x, right_tree)

data = pd.read_csv('CTG.csv', skiprows=1, usecols=range(1, 23))
data.dropna(inplace=True)

X, Y = data.iloc[:, :-2].values, data.iloc[:, -1].values
np.random.seed(42)

indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X, Y = X[indices], Y[indices]

split_idx = int(np.ceil(2/3 * X.shape[0]))
Xtrain, Xvalid = X[:split_idx], X[split_idx:]
Ytrain, Yvalid = Y[:split_idx], Y[split_idx:]
mean_values = np.mean(Xtrain, axis=0)
Xtrain_binary, Xvalid_binary = (Xtrain >= mean_values).astype(int), (Xvalid >= mean_values).astype(int)

class myDT:
    def __init__(self, Xtrain, Ytrain, Xvalid):
        self.Xtrain, self.Ytrain, self.Xvalid = Xtrain, Ytrain, Xvalid

    def fit(self):
        self.dt = DecisionTree(max_depth=5)
        self.dt.fit(self.Xtrain, self.Ytrain)

    def predict(self):
        return self.dt.predict(self.Xvalid)

dt_model = myDT(Xtrain_binary, Ytrain, Xvalid_binary)
dt_model.fit()
predictions = dt_model.predict()
accuracy = np.sum(predictions == Yvalid) / len(Yvalid)
confusion = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        confusion[i, j] = np.sum((predictions == i) & (Yvalid == j))

print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(confusion)
