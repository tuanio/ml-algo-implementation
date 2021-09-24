import numpy as np


class LinearRegression(object):
    def __init__(self, eta=0.01, n_iter=100, random_state=42):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.rs = np.random.RandomState(random_state)

    def fit(self, X, y):
        self.w = self.initialize_weights(X.shape[1] + 1)
        return self.partial_fit(X, y)

    def partial_fit(self, X, y):
        X = self.add_constant(X)
        for _ in range(self.n_iter):
            self._update_weights(X, y)
        return self

    def _update_weights(self, X, y):
        diff = self.predict(X) - y
        loss = self.eta * X.T.dot(diff)
        self.w -= loss

    def add_constant(self, X):
        ones = np.ones((X.shape[0], 1))
        return np.concatenate((ones, X), axis=1)

    def score(self, X, y):
        '''
        Calculate using R² Score
        '''
        predicted = self.predict(X)
        return 1 - (1 / X.shape[0]) * np.sum((y - predicted) ** 2 / (y - y.mean()) ** 2)

    def initialize_weights(self, size):
        return self.rs.normal(loc=0, scale=1, size=size)

    def predict(self, X):
        if X.shape[1] != self.w.shape[0]:
            X = self.add_constant(X)
        return self.w.dot(X.T)


class LinearRegressionSGD(LinearRegression):
    def __init__(self, eta=0.01, n_iter=100, random_state=42):
        super().__init__(eta, n_iter, random_state)

    def partial_fit(self, X, y):
        X = self.add_constant(X)
        for _ in range(self.n_iter):
            for xi, yi in zip(X, y):
                self._update_weights(xi[:, np.newaxis], yi)
        return self


class RidgeRegressionMiniBatchGD(LinearRegression):
    def __init__(self, eta=0.01, alpha=0.001, batch_size=32, n_iter=100, random_state=42):
        super().__init__(eta, n_iter, random_state)
        self.alpha = alpha
        self.batch_size = batch_size

    def partial_fit(self, X, y):
        X = self.add_constant(X)
        for _ in range(self.n_iter):
            X, y = self.shuffle(X, y)
            for idx in range(0, X.shape[0] - self.batch_size, self.batch_size):
                X_batch = X[idx:idx + self.batch_size]
                y_batch = y[idx:idx + self.batch_size]
                self._update_weights(X_batch, y_batch)
        return self

    def _update_weights(self, X, y):
        diff = self.predict(X) - y
        # ridge regression
        loss = self.eta * 2 * X.T.dot(diff) + 2 * \
            ((self.alpha * self.w) ** 2).sum()
        self.w -= loss

    def shuffle(self, X, y):
        r = self.rs.permutation(X.shape[0])
        return X[r], y[r]
