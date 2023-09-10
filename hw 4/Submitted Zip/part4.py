import numpy as np
import pandas as pd
from PIL import Image
import os
from part2 import myKNN
from Part3 import DecisionTree

yale_faces_root = 'yalefaces'
image_size = (40, 40)
yale_X = []
yale_Y = []

for person_id in os.listdir(yale_faces_root):
    person_folder = os.path.join(yale_faces_root, person_id)
    if os.path.isdir(person_folder):
        for image_name in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_name)
            image = Image.open(image_path).resize(image_size)  
            flattened_image = np.array(image).flatten()
            yale_X.append(flattened_image)
            yale_Y.append(int(person_id))

X = np.array(yale_X)
Y = np.array(yale_Y)

np.random.seed(42)
unique_persons = np.unique(Y)
Xtrain, Xvalid, Ytrain, Yvalid = [], [], [], []

for person_id in unique_persons:
    person_indices = np.where(Y == person_id)[0]
    np.random.shuffle(person_indices)
    split_idx = int(np.ceil(2/3 * len(person_indices)))
    train_indices = person_indices[:split_idx]
    valid_indices = person_indices[split_idx:]
    
    person_Xtrain = X[train_indices]  
    person_Xtrain_reshaped = person_Xtrain.reshape(-1, 1600)  
    Xtrain.append(person_Xtrain_reshaped)
    
    Xvalid.extend(X[valid_indices])
    Ytrain.extend(Y[train_indices])
    Yvalid.extend(Y[valid_indices])

Xtrain = np.concatenate(Xtrain, axis=0)
Xvalid = np.array(Xvalid)
Ytrain = np.array(Ytrain)
Yvalid = np.array(Yvalid)

k_values = [2, 9, 12]
yale_results_knn = []

for k in k_values:
    predictions_knn = myKNN(Xtrain, Ytrain, Xvalid, k)
    accuracy_knn = np.sum(predictions_knn == Yvalid) / len(Yvalid)
    confusion_knn = np.zeros((len(unique_persons), len(unique_persons)), dtype=int)
    for i in range(len(unique_persons)):
        for j in range(len(unique_persons)):
            confusion_knn[i, j] = np.sum((predictions_knn == unique_persons[i]) & (Yvalid == unique_persons[j]))
    yale_results_knn.append((k, accuracy_knn, confusion_knn))

yale_results_dt = []

for k in k_values:
    dt_model = DecisionTree(max_depth=5)
    dt_model.fit(Xtrain, Ytrain)
    predictions_dt = dt_model.predict(Xvalid)
    accuracy_dt = np.sum(predictions_dt == Yvalid) / len(Yvalid)
    confusion_dt = np.zeros((len(unique_persons), len(unique_persons)), dtype=int)
    for i in range(len(unique_persons)):
        for j in range(len(unique_persons)):
            confusion_dt[i, j] = np.sum((predictions_dt == unique_persons[i]) & (Yvalid == unique_persons[j]))
    yale_results_dt.append((k, accuracy_dt, confusion_dt))

print("Yale Faces Results - KNN:")
print("k\tAccuracy\tConfusion Matrix")
for k, accuracy, confusion in yale_results_knn:
    print(f"{k}\t{accuracy:.4f}\t{confusion}")

print("\nYale Faces Results - Decision Tree:")
print("k\tAccuracy\tConfusion Matrix")
for k, accuracy, confusion in yale_results_dt:
    print(f"{k}\t{accuracy:.4f}\t{confusion}")
