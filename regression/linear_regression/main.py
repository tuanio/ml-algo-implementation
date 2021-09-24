from linear_regression import RidgeLinearRegressionSGD
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes, load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def f(x, slope, intercept):
    return x * slope + intercept

slope = 5
intercept = 3

X = 3 * np.random.rand(2000, 1)
y = f(X[:, 0], slope, intercept) + np.random.rand(2000,) - 0.1

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=12)
lr = RidgeLinearRegressionSGD(eta=0.00001, alpha=0.0001, batch_size=64, n_iter=2000, random_state=12)
lr.fit(Xtrain, ytrain)

train_predicted = lr.predict(Xtrain)
test_predicted = lr.predict(Xtest)

r2_train = lr.score(Xtrain, ytrain)
r2_test = lr.score(Xtest, ytest)
print(r2_train, r2_test)
print(lr.w)

# rmse_train = mean_squared_error(ytrain, train_predicted, squared=False)
# rmse_test = mean_squared_error(ytest, test_predicted, squared=True)
# print(rmse_train, rmse_test)

plt.scatter(Xtrain, ytrain, label='train')
plt.scatter(Xtest, ytest, label='test')
predicted = lr.predict(Xtest)
plt.plot(Xtest, f(Xtest, slope, intercept), c='red', label='origin line')
plt.plot(Xtest, lr.predict(Xtest), c='yellow', label='model line')
plt.legend(loc='best')
plt.show()