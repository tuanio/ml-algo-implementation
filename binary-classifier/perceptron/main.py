import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from perceptron import Perceptron
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA

random_state = 12

# prepare datasets
X, y = load_breast_cancer(return_X_y=True)
X = PCA(2).fit_transform(X)
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
    eta=0.001,
    n_iter=100,
    random_state=random_state,
)

perceptron = Perceptron(**kwargs)
perceptron.fit(Xtrain, ytrain)

# metrics
train_score = accuracy_score(perceptron.predict(Xtrain), ytrain)
test_score = accuracy_score(perceptron.predict(Xtest), ytest)

print('Train score: %f' % train_score)
print('Test score: %f' % test_score)

markers = ('o', 'x')
colors = ('red', 'blue')
cmap = ListedColormap(colors[:2])
resolution = 0.02

x_min_max = []
for i in range(2):
    x_min_max.append((X[:, i].min() - 1, X[:, i].max() + 1))

xx1, xx2 = np.meshgrid(*[np.arange(mn, mx) for mn, mx in x_min_max])
Z = perceptron.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
Z = Z.reshape(xx1.shape)
plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
plt.xlim(xx1.min(), xx1.max())
plt.ylim(xx2.min(), xx2.max())

# plot class samples
for idx, cl in enumerate(np.unique(y)):
    plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8,
                c=colors[idx], marker=markers[idx], label=cl, edgecolor='black')

plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend(loc='upper left')
plt.show()