import numpy as np


class LogisticRegression:
    def __init__(self, n_iter=100, eta=0.001, threshold=0.5, random_state=42):
        self.n_iter = n_iter
        self.eta = eta
        self.threshold = threshold
        self.random_state = random_state
        self.rs = np.random.RandomState(self.random_state)

    def fit(self, X, y):
        self.w = self.__initialize_weights(X.shape[1])
        X = self.__add_constant(X)

        for _ in range(self.n_iter):
            predicted = self.predict_proba(X)
            update = -self.eta * (predicted - y).dot(X)
            self.w += update

        return self

    def __initialize_weights(self, m):
        return self.rs.normal(loc=0, scale=1, size=m + 1)

    def __add_constant(self, X):
        one = np.ones(shape=(X.shape[0], 1))
        return np.concatenate((one, X), axis=1)

    def __affine(self, X, w):
        return X.dot(w)

    def __activation(self, z):
        '''
            Sigmoid function
        '''
        return 1 / (1 + np.exp(-z))

    def predict_proba(self, X):
        if X.shape[1] != self.w.shape[0]:
            X = self.__add_constant(X)
        return self.__activation(self.__affine(X, self.w))

    def predict(self, X):
        return np.where(self.predict_proba(X) >= self.threshold, 1, 0)

    def score(self, X, y):
        X = self.__add_constant(X)
        predicted = self.predict(X)
        return np.sum(predicted == y) / y.shape[0]

    def __repr__(self):
        return 'LogisticRegression({})'.format(self.__dict__)
