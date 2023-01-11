import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from dataclasses import dataclass

@dataclass
class PCA:
    k: int = 2 # number of reduce dimension
    random_state: int = 12

    def fit_transform(self, X):
        X_norm = self.normalize(X)
        X_cov = self.get_covariance(X_norm)
        self.eigen_value, self.eigen_vector = self.get_eigen(X_cov)
        data = sorted(zip(self.eigen_value, self.eigen_vector.T), reverse=True)
        self.eigen_value = np.array([i[0] for i in data])
        self.eigen_vector = np.array([i[1] for i in data])
        W = np.stack(self.eigen_vector[:self.k], axis=1)
        return X_norm.dot(W)

    def normalize(self, X):
        return StandardScaler().fit_transform(X)
        # return X - X.mean()
    
    def get_covariance(self, X):
        return np.cov(X.T)
    
    def get_eigen(self, X):
        return np.linalg.eig(X)

    