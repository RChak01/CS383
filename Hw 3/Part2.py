import numpy as np
import os

data = np.genfromtxt('./spambase.data', delimiter=',')

np.random.seed(42)
np.random.shuffle(data)

# Columns
X = data[:, :-1]
y = data[:, -1]

split_idx = int(np.ceil((2/3) * X.shape[0]))

X_train = X[:split_idx]
X_val = X[split_idx:]
y_train = y[:split_idx]
y_val = y[split_idx:]

# Z Score
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
x_train_standard = (X_train - mean) / std
x_val_standard = (X_val - mean) / std

class_0_indices = np.where(y_train == 0)
class_1_indices = np.where(y_train == 1)

mean_class_0 = np.mean(x_train_standard[class_0_indices], axis=0)
mean_class_1 = np.mean(x_train_standard[class_1_indices], axis=0)

# Optimal Direction
optimal_direction = mean_class_1 - mean_class_0

X_val_projected = np.dot(x_val_standard, optimal_direction)

y_pred = np.where(X_val_projected >= 0, 1, 0)

accuracy = np.mean(y_pred == y_val)

TruePos = np.sum(np.logical_and(y_pred == 1, y_val == 1))
falPos = np.sum(np.logical_and(y_pred == 1, y_val == 0))
falNeg = np.sum(np.logical_and(y_pred == 0, y_val == 1))

precision = TruePos / (TruePos + falPos)
recall = TruePos / (TruePos + falNeg)
f_measure = 2 * (precision * recall) / (precision + recall)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F-Measure:", f_measure)
