from linear_regression import (
    RidgeRegressionMiniBatchGD,
    LinearRegressionSGD,
    LinearRegression
)
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes, load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.style.use('seaborn')


def f(x, slope, intercept):
    return x * slope + intercept


N = 500
slope = 5
intercept = 3
random_state = 35

rs = np.random.RandomState(random_state)

X = 3 * rs.rand(N, 1)
y = f(X[:, 0], slope, intercept) + rs.rand(N,) - 0.1

X = np.array([147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183])[:, np.newaxis]
y = np.array([ 49, 50, 51,  54, 58, 59, 60, 62, 63, 64, 66, 67, 68])

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=random_state)

lr_params = dict(eta=0.000001, n_iter=500, random_state=random_state)
# ridge_params = dict(eta=0.0001, alpha=0.001,
#                     batch_size=32, n_iter=500, random_state=random_state)
# lr_sgd_params = dict(eta=0.0001, n_iter=500, random_state=random_state)

lr = LinearRegression(**lr_params)
# ridge_mini = RidgeRegressionMiniBatchGD(**ridge_params)
# lr_sgd = LinearRegressionSGD(**lr_sgd_params)

# lr.fit(Xtrain, ytrain)
lr.fit(X, y)
print(lr.w)
print(mean_squared_error(y, lr.predict(X)))
# ridge_mini.fit(Xtrain, ytrain)
# lr_sgd.fit(Xtrain, ytrain)


# def plot_line(ax, X, y, label, regressor):
#     r2_train = regressor.score(Xtrain, ytrain)
#     r2_test = regressor.score(Xtest, ytest)
#     ax.scatter(X, y, c='orange', label='datasets')
#     ax.plot(X, f(X, slope, intercept), color='purple',
#             label=f'${slope}x + {intercept}$ (origin)')
#     ax.plot(X, f(X, round(regressor.w[1], 2), round(regressor.w[0], 2)),
#             color='blue', label=f'${round(regressor.w[1], 2)}x + {round(regressor.w[0], 2)}$ (modeled)')
#     ax.legend(loc='best')
#     ax.set_title(label)
#     fontsize = 'small'
#     ax.text(1, 0.5, '$R^2 Train: {}$'.format(r2_train), fontsize=fontsize)
#     ax.text(1, 0, '$R^2 Test: {}$'.format(r2_test), fontsize=fontsize)
#     return ax


# fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 12))
# ax[0] = plot_line(ax[0], X, y, 'Linear Regression', lr)
# ax[1] = plot_line(ax[1], X, y, 'Ridge Regression MiniBatch', ridge_mini)
# ax[2] = plot_line(ax[2], X, y, 'Linear Regression SGD', lr_sgd)
# fig.tight_layout()
# plt.savefig('linear.png')