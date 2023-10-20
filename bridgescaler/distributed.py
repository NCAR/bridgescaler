import numpy as np
from copy import copy


class DStandardScaler(object):
    """
    Distributed version of StandardScaler. You can calculate this map-reduce style by running it on individual
    data files, return the fitted objects, and then sum them together to represent the full dataset.

    """
    def __init__(self):
        self.sum_x_ = None
        self.n_ = 0
        self.sum_x2_ = None
        self.fit_ = False
        self.x_columns_ = None

    def fit(self, x):
        if hasattr(x, "columns"):
            self.x_columns_ = x.columns
        else:
            self.x_columns_ = np.arange(x.shape[-1])
        self.n_ += x.shape[0]
        if not self.fit_:
            self.sum_x_ = np.sum(x, axis=-1)
            self.sum_x2_ = np.sum(x ** 2, axis=-1)
        else:
            assert x.shape[1] == self.sum_x_.shape[0], "New data has a different number of columns than current scaler"
            self.sum_x_ += np.sum(x, axis=-1)
            self.sum_x2_ += np.sum(x ** 2, axis=-1)

    def transform(self, x):
        assert x.shape[1] == self.sum_x_.shape[0], "New data has a different number of columns than current scaler"
        x_mean = self.sum_x_ / self.n_
        x_sd = np.sqrt(self.sum_x2_ / self.n_ - x_mean ** 2)
        x_transformed = (x - x_mean) / x_sd
        return x_transformed

    def inverse_transform(self, x):
        assert x.shape[1] == self.sum_x_.shape[0], "New data has a different number of columns than current scaler"
        x_mean = self.sum_x_ / self.n_
        x_sd = np.sqrt(self.sum_x2_ / self.n_ - x_mean ** 2)
        x_transformed = x * x_sd + x_mean
        return x_transformed

    def __add__(self, other):
        assert type(other) == type(DStandardScaler), "Input is not DStandardScaler"
        assert other.x_columns_ == self.x_columns_, "Scaler columns do not match."
        current = copy(self)
        current.n_ += other.n_
        current.sum_x_ += current.sum_x_
        current.sum_x2_ += current.sum_x2_
        return current










    
