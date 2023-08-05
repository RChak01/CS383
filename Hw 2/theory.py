import numpy as np
y = np.array([1, -4,1,3, 11, 5, 0, -1, -3, 1])
X = np.array([-2, -5, -3, 0, -8, -2, 1, 5, -1, 6])


X_bias = np.vstack((np.ones(X.shape), X)).T 

X_transpose_X = np.dot(X_bias.T, X_bias)
X_transpose_y = np.dot(X_bias.T, y)
X_train,y_train = np.dot(np.linalg.inv(X_transpose_X), X_transpose_y)

print(X_train,y_train)