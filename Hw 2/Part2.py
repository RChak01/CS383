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

data = []
with open("H:\Year 4 Summer\CS 383\Hw 2\insurance.csv", 'r') as csvfile: # change this
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

split_idx = int(2 * data.shape[0] / 3)
train_data, val_data = data[:split_idx], data[split_idx:]

xTr, yTr = train_data[:, :-1], train_data[:, -1]
X_val, y_val = val_data[:, :-1], val_data[:, -1]


theta = lin_reg(xTr.astype(float), yTr.astype(float))

predictionsTrain = predict(xTr.astype(float), theta)
val_predictions = predict(X_val.astype(float), theta)

rmseTrain = rmse(yTr.astype(float), predictionsTrain)
val_rmse = rmse(y_val.astype(float), val_predictions)

train_smape = smape(yTr.astype(float), predictionsTrain)
val_smape = smape(y_val.astype(float), val_predictions)

print("Training RMSE:", rmseTrain)
print("Validation RMSE:", val_rmse)

print("Training SMAPE:", train_smape)
print("Validation SMAPE:", val_smape)