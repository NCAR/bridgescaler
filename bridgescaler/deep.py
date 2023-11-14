import numpy as np


class DeepStandardScaler(object):
    """
    Calculate standard scaler scores on an arbitrarily dimensional dataset as long as the last dimension is
    the variable dimension.

    """
    def __init__(self):
        self.mean_ = None
        self.sd_ = None
        return

    def fit(self, x):
        self.mean_ = np.zeros(x.shape[-1], dtype=x.dtype)
        self.sd_ = np.zeros(x.shape[-1], dtype=x.dtype)
        for v in range(x.shape[-1]):
            self.mean_[v] = np.mean(x[..., v])
            self.sd_[v] = np.std(x[..., v], ddof=1)

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


class DeepQuantileTransformer(object):
    """
    Performs a quantile transform on N-dimensional arrays where the variable dimension is the last one.

    Attributes:
        n_quantiles: number of quantiles to calculate and store
        stochastic: When transforming to quantile space, whether to take the mean of the left and right interpolation values (False)
            or to pick a random point in between (True).
    """
    def __init__(self, n_quantiles=1000, stochastic=False):
        self.n_quantiles = n_quantiles
        self.stochastic = stochastic
        self.quantiles_ = None
        self.references_ = None
        self.fitted_ = False
        self.x_column_names_ = None

    def fit(self, x):
        if hasattr(x, "columns"):
            self.x_columns_ = x.columns
        else:
            self.x_columns_ = np.arange(x.shape[-1])
        self.quantiles_ = np.zeros((x.shape[-1], self.n_quantiles), dtype=x.dtype)
        self.references_ = np.linspace(0, 1, self.n_quantiles, endpoint=True)
        for v in range(x.shape[-1]):
            self.quantiles_[v] = np.nanquantile(x[..., v].ravel(), self.references_)
            self.quantiles_[v] = np.maximum.accumulate(self.quantiles_[v])
        return

    def transform(self, x):
        x_transformed = np.zeros(x.shape, dtype=x.dtype)
        for v in range(x.shape[-1]):
            x_transformed[..., v] = self._transform_col(x[..., v].ravel(), v).reshape(x[..., v].shape)
        return x_transformed

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, x):
        x_transformed = np.zeros(x.shape, dtype=x.dtype)
        for v in range(x.shape[-1]):
            x_transformed[..., v] = self._inverse_transform_col(x[..., v].ravel(), v).reshape(x[..., v].shape)
        return x_transformed

    def _transform_col(self, x_col, col_index):
        left_ref = np.interp(x_col, self.quantiles_[col_index], self.references_)
        right_ref = -np.interp(-x_col, -self.quantiles_[col_index][::-1], -self.references_[::-1])
        p = 0.5
        if self.stochastic:
            p = np.random.uniform(0, 1, x_col.size)
        return p * left_ref + (1 - p) * right_ref

    def _inverse_transform_col(self, x_col, col_index):
        transformed_col = np.interp(x_col, self.references_, self.quantiles_[col_index])
        return transformed_col
