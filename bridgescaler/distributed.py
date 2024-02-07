import numpy as np
from copy import copy
from pytdigest import TDigest


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
            xv = x.values
        else:
            self.x_columns_ = np.arange(x.shape[-1])
            xv = x
        if not self.fit_:
            self.n_ += xv.shape[0]
            self.mean_x_ = np.zeros(xv.shape[-1], dtype=xv.dtype)
            self.var_x_ = np.zeros(xv.shape[-1], dtype=xv.dtype)
            for i in range(xv.shape[-1]):
                self.mean_x_[i] = np.mean(xv[..., i])
                self.var_x_[i] = np.var(xv[..., i], ddof=1)
        else:
            assert x.shape[-1] == self.mean_x_.shape[0], "New data has a different number of columns than current scaler"
            # update derived from https://math.stackexchange.com/questions/2971315/how-do-i-combine-standard-deviations-of-two-groups
            for i in range(x.shape[-1]):
                new_mean = np.mean(xv[..., i])
                new_var = np.var(xv[..., i], ddof=1)
                new_n = xv.shape[0]
                combined_mean = (self.n_ * self.mean_x_[i] + x.shape[0] * new_mean) / (self.n_ + x.shape[0])
                weighted_var = ((self.n_ - 1) * self.var_x_[i] + (new_n - 1) * new_var) / (self.n_ + new_n - 1)
                var_correction = self.n_ * new_n * (self.mean_x_[i] - new_mean) ** 2 / (
                        (self.n_ + new_n) * (self.n_ + new_n - 1))
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
    """
    Distributed MinMaxScaler enables calculation of min and max of variables in datasets in parallel then combining
    the mins and maxes as a reduction step.

    """
    def ___init__(self):
        self.max_x_ = None
        self.min_x_ = None
        self.fit_ = False
        self.x_columns_ = None

    def fit(self, x):
        if hasattr(x, "columns"):
            self.x_columns_ = x.columns
            xv = x.values
        else:
            self.x_columns_ = np.arange(x.shape[-1])
            xv = x
        if not self.fit_:
            self.max_x_ = np.zeros(x.shape[-1])
            self.min_x_ = np.zeros(x.shape[-1])
            for i in range(x.shape[-1]):
                self.max_x_[i] = np.max(xv[..., i])
                self.min_x_[i] = np.min(xv[..., i])
            self.fit_ = True
        else:
            for i in range(x.shape[-1]):
                self.max_x_[i] = np.maximum(self.max_x_[i], np.max(xv[..., i]))
                self.min_x_[i] = np.minimum(self.min_x_[i], np.min(xv[..., i]))

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
        assert type(other) is DMinMaxScaler, "Input is not DMinMaxScaler"
        assert np.all(other.x_columns_ == self.x_columns_), "Scaler columns do not match."
        current = copy(self)
        current.max_x_ = np.maximum(self.max_x_, other.max_x_)
        current.min_x_ = np.minimum(self.min_x_, other.min_x_)
        return current


class DQuantileTransformer(object):
    """
    Distributed quantile transformer that uses the T-Digest algorithm to transform the input data into approximate
    quantiles. This class stores the centroids and counts from the T-Digest algorithm and calls pytdigest to
    perform centroid fitting and transforms of data when needed. DQuantileTransformer objects can be added together
    to create a combined quantile distribution, enabling map-reduce-style distributed calculations of quantiles
    across large number of files. The same object can also be fit multiple times to compile quantiles serially
    but with lower memory usage.
    """
    def __init__(self, max_merged_centroids=1000, x_columns=None, centroids=None):
        self.max_merged_centroids = max_merged_centroids
        self._fit = False
        self.x_columns = x_columns
        self.centroids = centroids
        return

    def fit(self, x, weight=None):
        if hasattr(x, "columns"):
            self.x_columns = x.columns
        else:
            self.x_columns = np.arange(x.shape[-1])
        if not self._fit:
            # The number of centroids may vary depending on the distribution of each input variable.
            # The extra spots at the end are padded with nans.
            self.centroids = np.ones((x.shape[-1], self.max_merged_centroids, 2)) * np.nan
            for i in range(x.shape[-1]):
                td_obj = TDigest.compute(x[..., i].ravel(), w=weight, compression=self.max_merged_centroids)
                td_obj.force_merge()
                self.centroids[i, :td_obj._num_merged] = td_obj.get_centroids()
        else:
            td_objs = self.to_digests()
            new_centroids = np.ones((x.shape[-1], self.max_merged_centroids, 2)) * np.nan
            for i, td_obj in enumerate(td_objs):
                new_td_obj = TDigest.compute(x[..., i].ravel(), w=weight, compression=self.max_merged_centroids)
                combined_td_obj = td_obj + new_td_obj
                combined_td_obj.force_merge()
                new_centroids[i, :combined_td_obj._num_merged] = combined_td_obj.get_centroids()
            self.centroids = new_centroids
        return

    def to_digests(self):
        """
        Converts the centroids for each variable in DQuantileTransformer to a list of TDigest objects.

        Returns:
            td_objs: list of TDigest objects
        """
        assert self._fit, "Must call fit() before calling to_digests()"
        td_objs = []
        for i in range(self.centroids.shape[0]):
            centroid_len = np.where(np.isnan(self.centroids[i, 0]))[0].min()
            td_objs.append(TDigest.of_centroids(self.centroids[i, :, centroid_len],
                                                compression=self.max_merged_centroids))
        return td_objs

    def to_centroids(self, td_objs):
        centroids = np.ones((len(td_objs), self.max_merged_centroids, 2)) * np.nan
        for i in range(len(td_objs)):
            centroids[i, :td_objs[i]._num_merged] = td_objs[i].get_centroids()
        return centroids


    def transform(self, x):
        assert self._fit, "Scaler has not been fit."
        assert x.shape[-1] == len(self.x_columns), "New data has a different number of columns than current scaler"
        td_objs = self.to_digests()
        x_transformed = np.zeros(x.shape, dtype=x.dtype)
        for i in range(x.shape[-1]):
            x_transformed[..., i] = td_objs[i].cdf(x[..., i])
        return x_transformed

    def inverse_transform(self, x):
        assert self._fit, "Scaler has not been fit."
        assert x.shape[-1] == len(self.x_columns), "New data has a different number of columns than current scaler"
        x_transformed = np.zeros(x.shape, dtype=x.dtype)
        td_objs = self.to_digests()
        for i in range(x.shape[-1]):
            x_transformed[..., i] = td_objs[i].inverse_cdf(x[..., i])
        return x_transformed

    def __add__(self, other):
        assert type(other) is DQuantileTransformer, "Adding mismatched scaler types."
        td_objs = self.to_digests()
        other_td_objs = other.to_digests()
        assert len(td_objs) == len(other_td_objs), "Number of variables in scalers do not match."
        combined_centroids = np.ones(self.centroids.shape) * np.nan
        for i in range(len(td_objs)):
            combined_td_obj = td_objs[i] + other_td_objs[i]
            combined_td_obj.force_merge()
            combined_centroids[i, :combined_td_obj._num_merged] = combined_td_obj.get_centroids()
        new_dquantile = DQuantileTransformer(max_merged_centroids=self.max_merged_centroids,
                                             x_columns=self.x_columns,
                                             centroids=combined_centroids)
        new_dquantile._fit = True
        return new_dquantile












    
