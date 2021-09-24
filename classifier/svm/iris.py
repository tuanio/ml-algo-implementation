import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def plot_decision_regions(X, y, classifier, filename, resolution=0.02):
    
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8,
                    c=colors[idx], marker=markers[idx], label=cl, edgecolor='black')
    plt.savefig('{}.png'.format(filename))


data = load_iris()
X = data.data[:, :2]
y = data.target

X_std = StandardScaler().fit_transform(X)
Xtrain, Xtest, ytrain, ytest = train_test_split(X_std, y, test_size=0.2, random_state=1)

svm = SVC(kernel='rbf', degree=1, random_state=1, gamma=0.45, C=0.8)
svm.fit(Xtrain, ytrain)

acc_train = svm.score(Xtrain, ytrain)
acc_test = svm.score(Xtest, ytest)
print(acc_train, acc_test)

plot_decision_regions(X_std, y, svm, 'svm_iris')
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.show()