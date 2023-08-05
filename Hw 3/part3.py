import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('./spambase.data', header=None)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Randomize 
np.random.seed(42)
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

split_ratio = 2/3
split_idx = int(np.ceil(split_ratio * X.shape[0]))
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

# z-scores
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
x_train_standard = (X_train - mean) / std
x_val_standard = (X_val - mean) / std

# bias
X_train_std_bias = np.hstack((np.ones((x_train_standard.shape[0], 1)), x_train_standard))
X_val_std_bias = np.hstack((np.ones((x_val_standard.shape[0], 1)), x_val_standard))

np.random.seed(42)
num_features = X_train_std_bias.shape[1]
weights = np.random.rand(num_features)

learning_rate = 0.01
num_epochs = 3000

epsilon = 1e-14

train_log_loss = []
val_log_loss = []

for epoch in range(num_epochs):
    logits = np.dot(X_train_std_bias, weights)
    predictions = 1 / (1 + np.exp(-logits))
    
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    
    #log loss training data
    train_loss = -np.mean(y_train * np.log(predictions) + (1 - y_train) * np.log(1 - predictions))
    train_log_loss.append(train_loss)
    
    gradient = np.dot(X_train_std_bias.T, (predictions - y_train)) / y_train.shape[0]
    
    weights -= learning_rate * gradient
    
    #  log loss validation data
    val_logits = np.dot(X_val_std_bias, weights)
    val_predictions = 1 / (1 + np.exp(-val_logits))
    
    val_predictions = np.clip(val_predictions, epsilon, 1 - epsilon)
    
    val_loss = -np.mean(y_val * np.log(val_predictions) + (1 - y_val) * np.log(1 - val_predictions))
    val_log_loss.append(val_loss)

# Classification
y_pred = np.where(val_predictions >= 0.5, 1, 0)

truPos = np.sum(np.logical_and(y_pred == 1, y_val == 1))
falPos = np.sum(np.logical_and(y_pred == 1, y_val == 0))
falNeg = np.sum(np.logical_and(y_pred == 0, y_val == 1))

precision = truPos / (truPos + falPos)
recall = truPos / (truPos + falNeg)
f_measure = 2 * (precision * recall) / (precision + recall)
accuracy = np.mean(y_pred == y_val)

plt.plot(range(num_epochs), train_log_loss, label='Training Log Loss')
plt.plot(range(num_epochs), val_log_loss, label='Validation Log Loss')
plt.xlabel('Epoch')
plt.ylabel('Log Loss')
plt.legend()
plt.title('Epoch vs Log Loss')
plt.show()

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F-Measure:", f_measure)
