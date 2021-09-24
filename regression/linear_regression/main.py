from linear_regression import LinearRegressionSGD
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes, load_boston
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

X = 3 * np.random.rand(2000, 1)
y = 5 * X[:, 0] + 3 + np.random.rand(2000,) - 0.1

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=12)
lr = LinearRegressionSGD(eta=0.00001, batch_size=64, n_iter=2000, random_state=12)
lr.fit(Xtrain, ytrain)

r2_train = lr.score(Xtrain, ytrain)
r2_test = lr.score(Xtest, ytest)
print(r2_train, r2_test)

print(lr.w)

plt.scatter(Xtrain, ytrain, label='train')
plt.scatter(Xtest, ytest, label='test')
predicted = lr.predict(Xtest)
plt.plot(Xtest, lr.predict(Xtest), c='yellow')
plt.legend(loc='best')
plt.show()