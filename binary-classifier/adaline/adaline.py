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