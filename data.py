import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression



def generateData(n=100, d=2, scale=5, seed=42):
    """np.random.seed(seed)
    weights = np.random.uniform(low=-4,high =4 , size=(d,1))
    X = np.random.rand(n, d)*scale
    noise = np.random.normal(0, 1, n).reshape(n,1)
    Y = np.dot(X,weights)
    X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                        test_size=0.2, shuffle=False)
    feature_means = np.mean(X_train,axis =0)
    X_train = X_train - feature_means
    X_test = X_test - feature_means

    return X_train, X_test, y_train.reshape(y_train.shape[0],1), y_test"""
    """np.random.seed(seed)
    X = np.random.rand(n, d)*scale
    Y = np.sin(np.dot(X,np.ones((d,1))))
    X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                        test_size=0.2, shuffle=False)
    feature_means = np.mean(X_train,axis = 0)
    X_train = X_train - feature_means
    X_test = X_test - feature_means

    return X_train, X_test, y_train.reshape(y_train.shape[0],1), y_test"""
    np.random.seed(seed)
    weights = np.random.rand(d)
    negative_indices = np.random.choice(d, 3, replace=False)
    weights[negative_indices] = -np.random.rand(3)
    weights = weights.reshape((d,1))

    X = np.random.rand(n, d)*scale
    noise = np.random.normal(0, 10, n).reshape(n,1)
    Y = np.dot(X,weights) + noise
    X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                        test_size=0.2, shuffle=False)
    feature_means = np.mean(X_train,axis =0)
    X_train = X_train - feature_means
    X_test = X_test - feature_means

    return X_train, X_test, y_train.reshape(y_train.shape[0],1), y_test, negative_indices
