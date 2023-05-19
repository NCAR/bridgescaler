import numpy as np
import pandas as pd


class StochasticQuantileTransformer(object):
    def __init__(self, n_quantiles=1000):
        self.n_quantiles = n_quantiles
        self.references_ = None
        self.quantiles_ = None
        self.n_features_in_ = None
        self.feature_names_in_ = None
        return

    def fit(self, x):
        self.references_ = np.linspace(0, 1, self.n_quantiles, endpoint=True)
        is_df = hasattr(x, "columns")
        if is_df:
            self.feature_names_in_ = x.columns
            self.n_features_in_ = x.columns.size
        else:
            self.n_features_in_ = x.shape[1]
        self.quantiles_ = np.zeros((self.n_quantiles, self.n_features_in_))
        for f in range(self.n_features_in_):
            if is_df:
                x_col = x.iloc[:, f]
            else:
                x_col = x[:, f]
            self.quantiles_[:, f] = np.nanquantile(x_col, self.references_)
        return

    def transform(self, x):
        is_df = hasattr(x, "columns")
        if is_df:
            transformed = pd.DataFrame(np.zeros(x.shape), columns=x.columns, index=x.index)
        else:
            transformed = np.zeros(x.shape, dtype=x.dtype)
        for f in range(self.n_features_in_):
            if is_df:
                x_col = x.iloc[:, f]
                transformed.iloc[:, f] = self._transform_col(x_col.values, f)
            else:
                x_col = x[:, f]
                transformed[:, f] = self._transform_col(x_col, f)
        return transformed

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, x):
        is_df = hasattr(x, "columns")
        if is_df:
            transformed = pd.DataFrame(np.zeros(x.shape), columns=x.columns, index=x.index)
        else:
            transformed = np.zeros(x.shape, dtype=x.dtype)
        for f in range(self.n_features_in_):
            if is_df:
                x_col = x.iloc[:, f]
                transformed.iloc[:, f] = self._inverse_transform_col(x_col.values, f)
            else:
                x_col = x[:, f]
                transformed[:, f] = self._inverse_transform_col(x_col, f)
        return transformed

    def _transform_col(self, x_col, col_index):
        q_left = np.searchsorted(self.quantiles_[:, col_index], x_col, side="left")
        q_right = np.searchsorted(self.quantiles_[:, col_index], x_col, side="right")
        left_refs = self.references_[q_left]
        right_refs = self.references_[np.minimum(q_right, self.n_quantiles - 1)]
        transformed_col = (right_refs - left_refs) * np.random.random(x_col.size) + left_refs
        return transformed_col

    def _inverse_transform_col(self, x_col, col_index):
        transformed_col = np.interp(x_col, self.references_, self.quantiles_[:, col_index])
        return transformed_col


