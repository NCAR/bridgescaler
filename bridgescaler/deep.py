import numpy as np


class DeepStandardScaler(object):
    def __init__(self):
        self.mean_ = None
        self.sd_ = None
        return

    def fit(self, x):
        self.mean_ = np.zeros(x.shape[-1], dtype=x.dtype)
        self.sd_ = np.zeros(x.shape[-1], dtype=x.dtype)
        for v in range(x.shape[-1]):
            self.mean_[v] = np.mean(x[..., v])
            self.sd_[v] = np.std(x[..., v])

    def transform(self, x):
        x_transformed = np.zeros(x.shape, dtype=x.dtype)
        for v in range(x.shape[-1]):
            x_transformed[..., v] = (x[..., v] - self.mean_[v]) / self.sd_[v]
        return x_transformed

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, x):
        x_inverse = np.zeros(x.shape, dtype=x.dtype)
        for v in range(x.shape[-1]):
            x_inverse[..., v] = x[..., v] * self.sd_[v] + self.mean_[v]
        return x_inverse


class DeepMinMaxScaler(object):
    def __init__(self):
        self.max_ = None
        self.min_ = None
        return

    def fit(self, x):
        self.max_ = np.zeros(x.shape[-1], dtype=x.dtype)
        self.min_ = np.zeros(x.shape[-1], dtype=x.dtype)
        for v in range(x.shape[-1]):
            self.max_[v] = np.max(x[..., v])
            self.min_[v] = np.min(x[..., v])

    def transform(self, x):
        x_transformed = np.zeros(x.shape, dtype=x.dtype)
        for v in range(x.shape[-1]):
            x_transformed[..., v] = (x[..., v] - self.min_[v]) / (self.max_[v] - self.min_[v])
        return x_transformed

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, x):
        x_inverse = np.zeros(x.shape, dtype=x.dtype)
        for v in range(x.shape[-1]):
            x_inverse[..., v] = x[..., v] * (self.max_[v] - self.min_[v]) + self.min_[v]
        return x_inverse
