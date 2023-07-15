from PIL import Image
import numpy as np
import os
from numpy import linalg as LA
import matplotlib.pyplot as plt

toPath = 'yalefaces'
SEPARATOR = os.sep
firstMatrix = []
std_data = []


def PComponents(data, k):
    cov = np.cov(data.T)
    d, v = LA.eig(cov)
    indices = np.argsort(d)[::-1][:k]
    max_vectors = v[:, indices]
    return max_vectors

def standardize(arr):
    mean = arr.mean()
    std = arr.std(ddof=1)
    new_arr = (arr - mean) / std
    return new_arr



files = os.listdir(toPath)
for filename in files:
    if not filename.endswith('.txt'):
        im = Image.open(os.path.join(toPath, filename)).resize((40, 40))
        arr = np.array(im).flatten()
        firstMatrix.append(arr)
        std_data.append(standardize(arr))

std_data = np.array(std_data)

w = PComponents(std_data, 2)
matrix_w = w.T

Z = np.dot(matrix_w, std_data.T)

plt.scatter(Z[0], Z[1])
plt.xlabel('P.C 1')
plt.ylabel('P.C 2')
plt.title('Two PC')
plt.show()