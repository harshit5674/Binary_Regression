import numpy as np
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.model_selection import train_test_split

def getCaliforniaHousingData():
    data = fetch_california_housing()
    X = np.asarray(data.data)
    y = np.asarray(data.target)
    X_train, X_test, y_train, y_test = train_test_split(X,y, 
                                                test_size=0.2, shuffle= False)
    return X_train, X_test, y_train.reshape(y_train.shape[0],1), y_test
    

def getDiabetes():
    data = load_diabetes()
    X = np.asarray(data['data'])
    y = np.asarray(data['target'])
    X_train, X_test, y_train, y_test = train_test_split(X,y, 
                                                test_size=0.2, shuffle= False)
    return X_train, X_test, y_train.reshape(y_train.shape[0],1), y_test