import argparse
from data import generateData
from itertools import product
import numpy as np
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
parser.add_argument('--lr', type = np.float32, default=0.001)

args = parser.parse_args()

X, y = generateData(args.n, args.d, args.scale)

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
# Algorithm Weights
def grad_(X,y,beta):
    y_hat = np.dot(X, beta.T)
    error = y - y_hat
    mse = np.square(error).mean()
    grad = - (1 / args.n) * np.dot(X.T, error)
    return grad

def proj(beta):
    return np.sign(beta)

beta = np.random.rand(1,args.d)
count = 0
while (not np.array_equal(best_weights, beta)) and count != args.count:
    count = count+1
    grad = grad_(X,y,beta).T
    beta = beta - args.lr*grad
    beta = proj(beta)

print("Iterations to converge "+str(count))
lossPGD = mean_squared_error(np.dot(X,beta.T),y)
print(lossPGD)