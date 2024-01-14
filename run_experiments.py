from itertools import product
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from data import generateData

n = 3000

"""
0-> Original 
1-> STE
2-> LazyProx
3-> Prox
4-> ProxSTE
"""

x0 = []
x1 = []
x2 = []
x3 = []
x4 = []
y1=[]
y2=[]
y3=[]
y4=[]
z0=[]
z3=[]
z4=[]

d = 14

lambda_ = 0.001

def Relu(x):
    return x * (x > 0)

def getLambda_(t):
    return lambda_*t

def grad_(X,y,beta):
    y_hat = np.dot(X, beta.T)
    error = y - y_hat
    mse = np.square(error).mean()
    grad = - (1 / n) * np.dot(X.T, error)
    return grad

def quantize(beta):
    return np.sign(beta)

def proj(beta, lambda_):
    return np.sign(beta) + np.multiply(
        np.sign(beta - np.sign(beta)),
        Relu(np.absolute(beta - np.sign(beta))- lambda_) 
    )


for i in range(0,100,1):
    print(i)
    X, X_test, y, y_test,neg = generateData(n, d, 5, i)
    weight_combinations = list(product([-1, 1], repeat=d))

    best_mse = float('inf')
    best_weights = np.asarray(weight_combinations[0],dtype=np.float32)

    for weights in weight_combinations:

        Y = np.dot(X, (np.asarray(weights, dtype=np.float32)).reshape(1,d).T)
        mse = mean_squared_error(Y,y)
        if mse < best_mse:
            best_mse = mse
            best_weights = np.asarray(weights,dtype=np.float32).reshape(1,d)

    lossBinary = mean_squared_error(np.dot(X, best_weights.T),y)

    #OLS
    reg = LinearRegression().fit(X, y)
    modified_coef = np.sign(reg.coef_)
    x0.append(np.linalg.norm(modified_coef-best_weights))
    z0.append(mean_squared_error(np.dot(X_test,modified_coef.T),y_test))
    print(np.sort(neg))
    print(best_weights)
    print(modified_coef)

    """#STE
    beta = np.zeros((1,d))
    count = 0
    prev_beta = beta
    best_error = float('inf')
    new_best_weights = beta
    while (not np.array_equal(best_weights, quantize(beta))) and count != 100000:
    #while count != 20000:
        count = count+1
        grad = grad_(X,y,quantize(beta)).T
        if np.linalg.norm(grad) == 0:
            break
        beta = beta - 0.01*grad
        if np.array_equal(prev_beta, beta):
            print("Converged")
            break
        prev_beta = beta
        mse2 = mean_squared_error(np.dot(X,quantize(beta.T)),y)
        best_error = min(mse2, best_error)
        if best_error == mse2:
            new_best_weights = quantize(beta)
    x1.append(np.linalg.norm(best_weights - beta))
    y1.append(count)
    print(count)"""

    #LazyProx
    """print("STE")
    beta = np.zeros((1,d))
    count = 0
    prev_beta = beta
    best_error = float('inf')
    new_best_weights = beta
    lambda_=0.001
    while (not np.array_equal(best_weights, quantize(beta))) and count != 500000:
    #while count != args.count:
        count = count+1
        grad = grad_(X,y,proj(beta,getLambda_(count+1))).T
        if np.linalg.norm(grad) == 0:
            break
        beta = beta - 0.01*grad
        if np.array_equal(prev_beta, beta):
            print("Converged")
            break
        prev_beta = beta
        mse2 = mean_squared_error(np.dot(X,quantize(beta.T)),y)
        best_error = min(mse2, best_error)
        if best_error == mse2:
            new_best_weights = quantize(beta)
    x2.append(np.linalg.norm(org - new_best_weights))
    y2.append(count)"""

    #Prox
    """beta = np.zeros((1,d))
    count = 0
    prev_beta = beta
    best_error = float('inf')
    new_best_weights = beta
    lambda_=0.001
    #while (not np.array_equal(best_weights, quantize(beta))) and count != 500000:
    while count != 500000:
        count = count+1
        grad = grad_(X,y,beta).T
        if np.linalg.norm(grad) == 0:
            break
        beta = beta - 0.01*grad
        beta = proj(beta, getLambda_(count+1))
        if np.array_equal(prev_beta, beta):
            print("Converged")
            break
        prev_beta = beta
        mse2 = mean_squared_error(np.dot(X,quantize(beta.T)),y)
        best_error = min(mse2, best_error)
        if best_error == mse2:
            new_best_weights = quantize(beta)

    x3.append(np.linalg.norm(best_weights - beta))
    y3.append(count)
    z3.append(mean_squared_error(np.dot(X_test,beta.T),y_test))

    #ProxSTE
    beta = np.zeros((1,d))
    count = 0
    prev_beta = beta
    best_error = float('inf')
    new_best_weights = beta
    lambda_=0.001
    #while (not np.array_equal(best_weights, quantize(beta))) and count != 500000:
    while count != 500000:
        count = count+1
        grad = grad_(X,y,quantize(beta)).T
        if np.linalg.norm(grad) == 0:
            break
        beta = beta - 0.01*grad
        beta = proj(beta, getLambda_(count+1))
        if np.array_equal(prev_beta, beta):
            print("Converged")
            break
        prev_beta = beta
        mse2 = mean_squared_error(np.dot(X,quantize(beta.T)),y)
        best_error = min(mse2, best_error)
        if best_error == mse2:
            new_best_weights = quantize(beta)

    x4.append(np.linalg.norm(best_weights - beta))
    y4.append(count)
    z4.append(mean_squared_error(np.dot(X_test,beta.T),y_test))


df = pd.DataFrame(list(zip(x0,z0,x3,y3,z3,x4,y4,z4)),
               columns =['x0','z0','x3','y3','z3','x4','y4','z4'])

df.to_csv('experimentsin.csv')"""