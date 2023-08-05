import numpy as np
import csv
import os

def predict(X, theta):
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    
    predictions = X @ theta
    return predictions

def lin_reg(xTr, yTr):
    xTr = np.hstack((np.ones((xTr.shape[0], 1)), xTr))
    
    theta = np.linalg.inv(xTr.T @ xTr) @ xTr.T @ yTr
    
    return theta


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))

def s_fold_cross_validation(X, y, S=5):
    fold_size = len(X) // S
    metrics = {'RMSE': [], 'SMAPE': []}
    
    for i in range(S):
        idxStart = i * fold_size
        end_idx = (i + 1) * fold_size if i < S - 1 else len(X)

        X_val_fold = X[idxStart:end_idx]
        y_val_fold = y[idxStart:end_idx]

        X_train_fold = np.concatenate([X[:idxStart], X[end_idx:]], axis=0)
        y_train_fold = np.concatenate([y[:idxStart], y[end_idx:]], axis=0)

        theta = lin_reg(X_train_fold.astype(float), y_train_fold.astype(float))
        val_predictions = predict(X_val_fold.astype(float), theta)

        fold_rmse = rmse(y_val_fold.astype(float), val_predictions)
        fold_smape = smape(y_val_fold.astype(float), val_predictions)

        metrics['RMSE'].append(fold_rmse)
        metrics['SMAPE'].append(fold_smape)

    return metrics

data = []
with open("H:\Year 4 Summer\CS 383\Hw 2\insurance.csv", 'r') as csvfile: #change this
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        data.append(row)
data = np.array(data)
data = np.where(data == "northwest", '3', data)
data = np.where(data == "southeast", '2', data)
data = np.where(data == "southwest", '1', data)
data = np.where(data == "female", '1', data)
data = np.where(data == "male", '0', data)
data = np.where(data == "northeast", '0', data)
data = np.where(data == "yes", '1', data)
data = np.where(data == "no", '0', data)
data= np.delete(data,0,0)

np.random.seed(42)
np.random.shuffle(data)

X, y = data[:, :-1], data[:, -1]

num_runs = 20
S = 1338
avg_metrics = {'RMSE': 0.0, 'SMAPE': 0.0}

for _ in range(num_runs):
    metrics = s_fold_cross_validation(X.astype(float), y.astype(float), S)
    avg_metrics['RMSE'] += np.mean(metrics['RMSE'])
    avg_metrics['SMAPE'] += np.mean(metrics['SMAPE'])

avg_metrics['RMSE'] /= num_runs
avg_metrics['SMAPE'] /= num_runs

print(f"Avg. RMSE for 3-fold cross-validation over {num_runs} runs:", avg_metrics['RMSE'])
print(f"Avg. SMAPE for 3-fold cross-validation over {num_runs} runs:", avg_metrics['SMAPE'])