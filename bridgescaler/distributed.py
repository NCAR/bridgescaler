import numpy as np
from copy import copy


class DStandardScaler(object):
    """
    Distributed version of StandardScaler. You can calculate this map-reduce style by running it on individual
    data files, return the fitted objects, and then sum them together to represent the full dataset.

    """
    def __init__(self):
        self.mean_x_ = None
        self.n_ = 0
        self.var_x_ = None
        self.fit_ = False
        self.x_columns_ = None

    def fit(self, x):
        if hasattr(x, "columns"):
            self.x_columns_ = x.columns
        else:
            self.x_columns_ = np.arange(x.shape[-1])
        if not self.fit_:
            self.n_ += x.shape[0]
            self.mean_x_ = np.zeros(x.shape[-1], dtype=x.dtype)
            self.var_x_ = np.zeros(x.shape[-1], dtype=x.dtype)
            for i in range(x.shape[-1]):
                self.mean_x_[i] = np.mean(x[..., i])
                self.var_x_[i] = np.var(x[..., i], ddof=1)
        else:
            assert x.shape[-1] == self.mean_x_.shape[0], "New data has a different number of columns than current scaler"
            # update derived from https://math.stackexchange.com/questions/2971315/how-do-i-combine-standard-deviations-of-two-groups
            for i in range(x.shape[-1]):
                new_mean = np.mean(x[..., i])
                new_var = np.var(x[..., i], ddof=1)
                new_n = x.shape[0]
                combined_mean = (self.n_ * self.mean_x_ + x.shape[0] * new_mean) / (self.n_ + x.shape[0])
                weighted_var = ((self.n_ - 1) * self.var_x_[i] + (new_n - 1) * new_var) / (self.n_ + new_n - 1)
                var_correction = self.n_ * new_n * (self.mean_x_[i] - new_mean) ** 2 / ((self.n_ + new_n) * (self.n_ + new_n - 1))
                self.mean_x_[i] = combined_mean
                self.var_x_[i] = weighted_var + var_correction
                self.n_ += new_n

    def transform(self, x):
        assert self.fit_, "Scaler has not been fit."
        assert x.shape[-1] == self.mean_x_.shape[0], "New data has a different number of columns than current scaler"
        x_mean, x_var = self.get_scales()
        x_transformed = (x - x_mean) / np.sqrt(x_var)
        return x_transformed

    def inverse_transform(self, x):
        assert self.fit_, "Scaler has not been fit."
        assert x.shape[1] == self.mean_x_.shape[0], "New data has a different number of columns than current scaler"
        x_mean, x_var = self.get_scales()
        x_transformed = x * np.sqrt(x_var) + x_mean
        return x_transformed

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def get_scales(self):
        return self.mean_x_, self.var_x_

    def __add__(self, other):
        assert type(other) == DStandardScaler, "Input is not DStandardScaler"
        assert np.all(other.x_columns_ == self.x_columns_), "Scaler columns do not match."
        current = copy(self)
        current.mean_x_ = (self.n_ * self.mean_x_ + other.n_ * other.mean_x_) / (self.n_ + other.n_)
        combined_var = ((self.n_ - 1) * self.var_x_ + (other.n_ - 1) * other.var_x_) / (self.n_ + other.n_ - 1)
        combined_var_corr = self.n_ * other.n_ * (self.mean_x_ - other.mean_x_) ** 2 / ((self.n_ + other.n_) * (self.n_ + other.n_ - 1))
        current.var_x_ = combined_var + combined_var_corr
        current.n_ = self.n_ + other.n_
        return current


class DMinMaxScaler(object):

    def ___init__(self):
        self.max_x_ = None
        self.min_x_ = None
        self.fit_ = False
        self.x_columns_ = None

    def fit(self, x):
        if hasattr(x, "columns"):
            self.x_columns_ = x.columns
        else:
            self.x_columns_ = np.arange(x.shape[-1])
        if not self.fit_:
            self.max_x_ = np.zeros(x.shape[-1])
            self.min_x_ = np.zeros(x.shape[-1])
            for i in range(x.shape[-1]):
                self.max_x_[i] = np.max(x[..., i])
                self.min_x_[i] = np.min(x[..., i])
            self.fit_ = True
        else:
            for i in range(x.shape[-1]):
                self.max_x_[i] = np.maximum(self.max_x_[i], np.max(x[..., i]))
                self.min_x_[i] = np.minimum(self.min_x_[i], np.min(x[..., i]))

    def transform(self, x):
        assert self.fit_, "Scaler has not been fit."
        assert x.shape[-1] == self.min_x_.shape[0], "New data has a different number of columns than current scaler"
        x_transformed = (x - self.min_x_) / (self.max_x_ - self.min_x_)
        return x_transformed

    def inverse_transform(self, x):
        assert self.fit_, "Scaler has not been fit."
        assert x.shape[-1] == self.min_x_.shape[0], "New data has a different number of columns than current scaler"
        x_transformed = x * (self.max_x_ - self.min_x_) + self.min_x_
        return x_transformed

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def __add__(self, other):
        assert type(other) == DMinMaxScaler, "Input is not DMinMaxScaler"
        assert np.all(other.x_columns_ == self.x_columns_), "Scaler columns do not match."
        current = copy(self)
        current.max_x_ = np.maximum(self.max_x_, other.max_x_)
        current.min_x_ = np.minimum(self.min_x_, other.min_x_)
        return current













    
