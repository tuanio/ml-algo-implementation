import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from adaline import Adaline
import matplotlib.pyplot as plt

random_state = 12

# prepare datasets
X, y = load_breast_cancer(return_X_y=True)
y[y == 0] = -1  # to make two class, -1 and 1
Xy = np.concatenate((X, y.reshape(-1, 1)), axis=1)

# scale all features to the same range
std_scaler = StandardScaler()
X = std_scaler.fit_transform(X)

# split train and test
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=1/3, random_state=random_state)

# training phrase
kwargs = dict(
    eta=0.00001,
    n_iter=100,
    batch_size=64,
    random_state=random_state,
    verbose=0
)

adaline = Adaline(**kwargs)
adaline.fit(Xtrain, ytrain)

# metrics
train_score = accuracy_score(adaline.predict(Xtrain), ytrain)
test_score = accuracy_score(adaline.predict(Xtest), ytest)

print('Train score: %f' % train_score)
print('Test score: %f' % test_score)

# plot the prediction
# just use two highest features
mx = np.argsort(adaline.w_)[:2]

red_points = Xy[Xy[:, -1] == -1]
blue_points = Xy[Xy[:, -1] == 1]

plt.scatter(red_points[:, mx[0]], red_points[:, mx[1]], c='red')
plt.scatter(blue_points[:, mx[0]], blue_points[:, mx[1]], c='blue')
plt.show()
