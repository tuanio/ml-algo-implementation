import time
import sys
import numpy as np


class Adaline(object):
    def __init__(self, eta=0.01, n_iter=50, batch_size=32, random_state=1, verbose=0):
        self.eta = eta
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        # add a one column to X
        ones = np.ones(shape=(X.shape[0], 1))
        X = np.concatenate((ones, X), axis=1)

        # re-initialize w
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(size=X.shape[1])

        # concat X and y
        y = y.reshape(-1, 1)
        Xy = np.concatenate((X, y), axis=1)

        for _ in range(self.n_iter):
            np.random.shuffle(Xy)

            if self.verbose:
                sys.stdout.write(f'Epoch {_ + 1}/{self.n_iter}\n')

            idx = 0
            end = int(np.ceil(X.shape[0] / self.batch_size))

            for i in range(0, X.shape[0], self.batch_size):
                idx += 1

                start, stop = i, i + self.batch_size
                Xbatch, ybatch = Xy[start:stop, :-1], Xy[start:stop, -1]

                z = self.output(Xbatch)
                delta_w = - self.eta * \
                    (ybatch - z).dot(Xbatch) / ybatch.shape[0]
                self.w_ -= delta_w

                if self.verbose:
                    done = '*' * (idx)
                    not_done = '.' * (end - idx)
                    sys.stdout.write(f"{idx}/{end}: [{done}{not_done}]\r")

            if self.verbose:
                sys.stdout.write('\n')

        return self

    def output(self, X):
        if X.shape[1] + 1 == self.w_.shape[0]:
            ones = np.ones(shape=(X.shape[0], 1))
            X = np.concatenate((ones, X), axis=1)
        return X.dot(self.w_)

    def predict(self, X):
        return np.where(self.output(X) >= 0, 1, -1)


class AdalineSGD(object):
    """ADAptive LInear NEuron classifier.
    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    shuffle : bool (default: True)
      Shuffles training data every epoch if True 
      to prevent cycles.
    random_state : int
      Random number generator seed for random weight
      initialization.
    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
    cost_ : list Sum-of-squares cost function value averaged over all
      training samples in each epoch.
    """

    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.shuffle = shuffle
        self.random_state = random_state

    def fit(self, X, y):
        self._initialize_weights(X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)
        return self

    def partial_fit(self, X, y):
        """Fit training data without reinitializing the weights"""
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def _shuffle(self, X, y):
        """Shuffle training data"""
        r = self.rgen.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):
        """Initialize weights to small random numbers"""
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=1 + m)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        """Apply Adaline learning rule to update the weights"""
        output = self.activation(self.net_input(xi))
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error**2
        return cost

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return X

    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)
