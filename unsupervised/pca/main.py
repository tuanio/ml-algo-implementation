from pca import PCA
from sklearn import decomposition
from sklearn.datasets import load_iris, load_breast_cancer, load_wine
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

pca = PCA(k=2)
print(pca)

X, y = load_wine(return_X_y=True)
y = y[:, np.newaxis]

data = pca.fit_transform(X)
w_1 = np.array(sorted(pca.eigen_value, reverse=True))
w_1 /= w_1.sum()
vis_data = np.concatenate((data, y), axis=1)

# df = pd.DataFrame(vis_data, columns=['pca_1', 'pca_2', 'label'])
# sns.scatterplot(data=df, x='pca_1', y='pca_2', hue='label')
# plt.savefig('mypca.png')
# plt.show()

pca2 = decomposition.PCA(n_components=None)
X_fited = pca2.fit_transform(X)
# vis_data_2 = np.concatenate((X_fited, y), axis=1)
# df2 = pd.DataFrame(vis_data_2, columns=['pca_1', 'pca_2', 'label'])
# sns.scatterplot(data=df2, x='pca_1', y='pca_2', hue='label')
# plt.savefig('sklearnpca.png')
# plt.show()

w_2 = pca2.explained_variance_ratio_

print(w_1)
print(w_2)  

# w_cumsum = np.cumsum(w_2)
x_plot = np.arange(len(w_2))
plt.plot(x_plot, w_1, label='mypca')
plt.scatter(x_plot, w_1, label='mypca')
plt.plot(x_plot, w_2, label='sklearn')
plt.scatter(x_plot, w_2, label='sklearn')
plt.legend(loc='best')
plt.show()
# plt.bar(x_plot, w_2)
# plt.step(x_plot, w_cumsum)
# plt.show()
