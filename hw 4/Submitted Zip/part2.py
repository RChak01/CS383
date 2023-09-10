import numpy as np
import pandas as pd

def myKNN(Xtrain, Ytrain, Xvalid, k):
    predictions = []

    for i in range(Xvalid.shape[0]):
        distances = np.sqrt(np.sum((Xtrain - Xvalid[i])**2, axis=1))
        sorted_indices = np.argsort(distances)
        k_nearest_indices = sorted_indices[:k]
        k_nearest_labels = Ytrain[k_nearest_indices]
        unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)
        predicted_label = unique_labels[np.argmax(counts)]
        predictions.append(predicted_label)

    return np.array(predictions)

data = pd.read_csv('CTG.csv', skiprows=1, usecols=range(1, 23))

data.dropna(inplace=True)

X = data.iloc[:, :-2].values  
Y = data.iloc[:, -1].values   

np.random.seed(42)
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]

split_idx = int(np.ceil(2/3 * X.shape[0]))
Xtrain, Xvalid = X[:split_idx], X[split_idx:]
Ytrain, Yvalid = Y[:split_idx], Y[split_idx:]

Xtrain_mean = np.mean(Xtrain, axis=0)
Xtrain_std = np.std(Xtrain, axis=0)
Xtrain = (Xtrain - Xtrain_mean) / Xtrain_std
Xvalid = (Xvalid - Xtrain_mean) / Xtrain_std

k_values = [2, 9, 12]

results = []

for k in k_values:
    predictions = myKNN(Xtrain, Ytrain, Xvalid, k)
    
    accuracy = np.sum(predictions == Yvalid) / len(Yvalid)
    
    confusion = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            confusion[i, j] = np.sum((predictions == i) & (Yvalid == j))
    
    results.append((k, accuracy, confusion))

print("k\tAccuracy\tConfusion Matrix")
for k, accuracy, confusion in results:
    print(f"{k}\t{accuracy:.4f}\t{confusion}")
