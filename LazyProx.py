import argparse
from data import generateData
from dataset import getCaliforniaHousingData, getDiabetes
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

parser = argparse.ArgumentParser(description='Regression Parameters')

parser.add_argument('--n', type=int, default=100,
                    help='Number of training Samples')

parser.add_argument('--d', type=int, default= 1,
                    help='Dimensionality of Data')
parser.add_argument('--count', type=int, default = 100000,
                    help='Number of training Samples')
parser.add_argument('--scale', type=int, default=10,
                    help='Scale')
parser.add_argument('--lr', type = np.float32, default=0.01)
parser.add_argument('--reg', type = int, default=1)
parser.add_argument('--lambda_', type = np.float32, default=0.01)
parser.add_argument('--seed', type = int, default=42)

args = parser.parse_args()

lambda_ = args.lambda_

X, X_test, y, y_test  = generateData(args.n, args.d, args.scale, args.seed)


# Full precision weights
reg = LinearRegression(fit_intercept=False).fit(X,y)
fullWeights = reg.coef_
Y = reg.predict(X)
lossFullWeights = mean_squared_error(Y,y)
print(lossFullWeights)

# Best Quantized weights

weight_combinations = list(product([-1, 1], repeat=args.d))

best_mse = float('inf')
best_weights = np.asarray(weight_combinations[0],dtype=np.float32)

for weights in weight_combinations:

    Y = np.dot(X, (np.asarray(weights, dtype=np.float32)).reshape(1,args.d).T)
    mse = mean_squared_error(Y,y)
    if mse < best_mse:
        best_mse = mse
        best_weights = np.asarray(weights,dtype=np.float32).reshape(1,args.d)

lossBinary = mean_squared_error(np.dot(X, best_weights.T),y)
print("Binary Loss is "+str(lossBinary))
print(best_weights)
# Algorithm Weights

def quantize(x):
    return np.sign(x)

def Relu(x):
    return x * (x > 0)

def getLambda_(t):
    return lambda_*t

def grad_(X,y,beta):
    y_hat = np.dot(X, beta.T)
    error = y - y_hat
    mse = np.square(error).mean()
    grad = - (1 / args.n) * np.dot(X.T, error)
    return grad


def proj(beta, lambda_):
    if args.reg==1:
        return np.sign(beta) + np.multiply(
            np.sign(beta - np.sign(beta)),
            Relu(np.absolute(beta - np.sign(beta))- lambda_) 
        )
    else:
        return (beta + lambda_*np.sign(beta))/(1+lambda_)

beta = np.zeros((1,args.d))
prev_beta = beta
count = 0
xx = []
yy = []
best_error = float('inf')
new_best_weights = beta
while (not np.array_equal(best_weights, quantize(beta))) and count != args.count:
    count = count+1
    grad = grad_(X,y,proj(beta,getLambda_(count+1))).T
    if np.linalg.norm(grad) == 0:
        print("Grad_zero")
        break
    beta = beta - args.lr*grad
    if np.array_equal(prev_beta, beta):
        print("Converged")
        break
    prev_beta = beta
    mse1 = mean_squared_error(np.dot(X,beta.T),y)
    mse2 = mean_squared_error(np.dot(X,quantize(beta.T)),y)
    yy.append(mse1)
    xx.append(count)
    best_error = min(mse2, best_error)
    if best_error == mse2:
        new_best_weights = quantize(beta)

beta = new_best_weights
print("Iterations to converge "+str(count))
lossPGD = mean_squared_error(np.dot(X,beta.T),y)
plt.plot(xx,yy)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Prox')

from datetime import datetime
now = datetime.now()
current_time = now.strftime("%H:%M:%S")

plt.savefig('Results/LazyProx/'+str(args.seed)+'.png')
plt.show()
print("Training Loss "+str(best_error))
print("Test Loss "+str(mean_squared_error(np.dot(X_test,beta.T),y_test)))
print(new_best_weights)