import numpy as np


class Perceptron(object):
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1] + 1)
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for xi, yi in zip(X, y):
                yhat = self.predict(xi)
                delta_w = self.eta * (yi - yhat)
                self.w_[1:] += delta_w * xi
                self.w_[0] += delta_w
                errors += int(delta_w != 0)
            self.errors_.append(errors)
        return self

    def output(self, X):
        return X.dot(self.w_[1:]) + self.w_[0]

    def predict(self, X):
        # the sign function: sign(w.x)
        return np.where(self.output(X) >= 0.0, 1, -1)
