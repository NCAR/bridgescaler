import numpy as np
from copy import copy, deepcopy

from pytdigest import TDigest
from scipy.stats import norm, logistic
from xarray import DataArray
from pandas import DataFrame

class DBaseScaler(object):
    """
    Base distributed scaler class. Used only to store attributes and methods shared across all distributed
    scaler subclasses.
    """
    def __init__(self):
        self.x_columns_ = None
        self.is_array_ = False
        self._fit = False

    def is_fit(self):
        return self._fit

    @staticmethod
    def extract_x_columns(x):
        """
        Extract the variable names to be transformed from x depending on if x is a pandas DataFrame, an
        xarray DataArray, or a numpy array. All of these assume that the columns are in the last dimension.
        If x is an xarray DataArray, there should be a coorindate variable with the same name as the last dimension
        of the DataArray being transformed.

        Args:
            x (Union[pandas.DataFrame, xarray.DataArray, numpy.ndarray]): array of values to be transformed.

        Returns:
            xv (numpy.ndarray): Array of values to be transformed.
            is_array (bool): Whether or not x was a np.ndarray.
        """
        is_array = False
        if hasattr(x, "columns"):
            x_columns = x.columns
        elif hasattr(x, "coords"):
            var_dim = x.dims[-1]
            x_columns = x.coords[var_dim].values
        else:
            x_columns = np.arange(x.shape[-1])
            is_array = True
        return x_columns, is_array

    @staticmethod
    def extract_array(x):
        if hasattr(x, "columns") or hasattr(x, "coords"):
            xv = x.values
        else:
            xv = x
        return xv

    def get_column_order(self, x_in_columns):
        """
        Get the indices of the scaler columns that have the same name as the columns in the input x array. This
        enables users to pass a DataFrame or DataArray to transform or inverse_transform with fewer columns than
        the original scaler or columns in a different order and still have the input dataset be transformed properly.

        Args:
            x_in_columns (Union[list, numpy.ndarray]): list of input columns.

        Returns:
            x_in_col_indices (np.ndarray): indices of the input columns from x in the scaler in order.
        """
        assert np.all(np.isin(x_in_columns, self.x_columns_)), "Some input columns not in scaler x_columns."
        x_in_col_indices = np.array([np.where(col == np.array(self.x_columns_))[0][0] for col in x_in_columns])
        return x_in_col_indices
    
    @staticmethod
    def package_transformed_x(x_transformed, x):
        """
        Repackaged a transformed numpy array into the same datatype as the original x, including
        all metadata.

        Args:
            x_transformed (numpy.ndarray): array after being transformed or inverse transformed
            x (Union[pandas.DataFrame, xarray.DataArray, numpy.ndarray]):

        Returns:

        """
        if hasattr(x, "columns"):
            x_packaged = deepcopy(x)
            x_packaged.loc[:, :] = x_transformed
        elif hasattr(x, "coords"):
            x_packaged = deepcopy(x)
            x_packaged[:] = x_transformed
        else:
            x_packaged = x_transformed
        return x_packaged

    def fit(self, x, weight=None):
        pass

    def transform(self, x):
        pass

    def fit_transform(self, x):
        pass

    def inverse_transform(self, x):
        pass

    def __add__(self, other):
        pass

    def subset_columns(self, sel_columns):
        pass

    def add_variables(self, other):
        pass


class DStandardScaler(DBaseScaler):
    """
    Distributed version of StandardScaler. You can calculate this map-reduce style by running it on individual
    data files, return the fitted objects, and then sum them together to represent the full dataset. Scaler
    supports numpy arrays, pandas dataframes, and xarray DataArrays and will return a transformed array in the
    same form as the original with column or coordinate names preserved.

    """
    def __init__(self):
        self.mean_x_ = None
        self.n_ = 0
        self.var_x_ = None
        super().__init__()

    def fit(self, x, weight=None):
        self.x_columns_, self.is_array_ = self.extract_x_columns(x)
        xv = self.extract_array(x)
        if not self._fit:
            self.n_ += xv.shape[0]
            self.mean_x_ = np.zeros(xv.shape[-1], dtype=xv.dtype)
            self.var_x_ = np.zeros(xv.shape[-1], dtype=xv.dtype)
            for i in range(xv.shape[-1]):
                self.mean_x_[i] = np.nanmean(xv[..., i])
                self.var_x_[i] = np.nanvar(xv[..., i], ddof=1)
        else:
            assert x.shape[-1] == self.mean_x_.shape[0], "New data has a different number of columns"
            # update derived from https://math.stackexchange.com/questions/2971315/how-do-i-combine-standard-deviations-of-two-groups
            for i in range(x.shape[-1]):
                new_mean = np.nanmean(xv[..., i])
                new_var = np.nanvar(xv[..., i], ddof=1)
                new_n = xv.shape[0]
                combined_mean = (self.n_ * self.mean_x_[i] + x.shape[0] * new_mean) / (self.n_ + x.shape[0])
                weighted_var = ((self.n_ - 1) * self.var_x_[i] + (new_n - 1) * new_var) / (self.n_ + new_n - 1)
                var_correction = self.n_ * new_n * (self.mean_x_[i] - new_mean) ** 2 / (
                        (self.n_ + new_n) * (self.n_ + new_n - 1))
                self.mean_x_[i] = combined_mean
                self.var_x_[i] = weighted_var + var_correction
                self.n_ += new_n
        self._fit = True

    def transform(self, x):
        assert self._fit, "Scaler has not been fit."
        x_in_cols, is_array = self.extract_x_columns(x)
        if is_array:
            assert x.shape[-1] == self.mean_x_.shape[0], "Number of input columns does not match scaler."
            x_col_order = np.arange(x.shape[-1])
        else:
            x_col_order = self.get_column_order(x_in_cols)
        x_mean, x_var = self.get_scales()
        x_transformed = (x - x_mean[x_col_order]) / np.sqrt(x_var[x_col_order])
        x_transformed = self.package_transformed_x(x_transformed, x)
        return x_transformed

    def inverse_transform(self, x):
        assert self._fit, "Scaler has not been fit."
        x_in_cols, is_array = self.extract_x_columns(x)
        if is_array or self.is_array_:
            assert x.shape[-1] == self.mean_x_.shape[0], "Number of input columns does not match scaler."
            x_col_order = np.arange(x.shape[-1])
        else:
            x_col_order = self.get_column_order(x_in_cols)
        x_mean, x_var = self.get_scales()
        x_transformed = x * np.sqrt(x_var[x_col_order]) + x_mean[x_col_order]
        x_transformed = self.package_transformed_x(x_transformed, x)
        return x_transformed

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def get_scales(self):
        return self.mean_x_, self.var_x_

    def __add__(self, other):
        assert type(other) is DStandardScaler, "Input is not DStandardScaler"
        assert np.all(other.x_columns_ == self.x_columns_), "Scaler columns do not match."
        current = deepcopy(self)
        current.mean_x_ = (self.n_ * self.mean_x_ + other.n_ * other.mean_x_) / (self.n_ + other.n_)
        combined_var = ((self.n_ - 1) * self.var_x_ + (other.n_ - 1) * other.var_x_) / (self.n_ + other.n_ - 1)
        combined_var_corr = self.n_ * other.n_ * (self.mean_x_ - other.mean_x_) ** 2 / (
                (self.n_ + other.n_) * (self.n_ + other.n_ - 1))
        current.var_x_ = combined_var + combined_var_corr
        current.n_ = self.n_ + other.n_
        return current


class DMinMaxScaler(DBaseScaler):
    """
    Distributed MinMaxScaler enables calculation of min and max of variables in datasets in parallel then combining
    the mins and maxes as a reduction step. Scaler
    supports numpy arrays, pandas dataframes, and xarray DataArrays and will return a transformed array in the
    same form as the original with column or coordinate names preserved.

    """
    def __init__(self):
        self.max_x_ = None
        self.min_x_ = None
        super().__init__()

    def fit(self, x, weight=None):
        self.x_columns_, self.is_array_ = self.extract_x_columns(x)
        xv = self.extract_array(x)
        if not self._fit:
            self.max_x_ = np.zeros(xv.shape[-1])
            self.min_x_ = np.zeros(xv.shape[-1])
            for i in range(xv.shape[-1]):
                self.max_x_[i] = np.nanmax(xv[..., i])
                self.min_x_[i] = np.nanmin(xv[..., i])

        else:
            for i in range(x.shape[-1]):
                self.max_x_[i] = np.maximum(self.max_x_[i], np.nanmax(xv[..., i]))
                self.min_x_[i] = np.minimum(self.min_x_[i], np.nanmin(xv[..., i]))
        self._fit = True

    def transform(self, x):
        assert self._fit, "Scaler has not been fit."
        x_in_cols, is_array = self.extract_x_columns(x)
        if is_array:
            assert x.shape[-1] == self.max_x_.shape[0], "Number of input columns does not match scaler."
            x_col_order = np.arange(x.shape[-1])
        else:
            x_col_order = self.get_column_order(x_in_cols)
        x_transformed = (x - self.min_x_[x_col_order]) / (self.max_x_[x_col_order] - self.min_x_[x_col_order])
        x_transformed = self.package_transformed_x(x_transformed, x)
        return x_transformed

    def inverse_transform(self, x):
        assert self._fit, "Scaler has not been fit."
        x_in_cols, is_array = self.extract_x_columns(x)
        if is_array:
            assert x.shape[-1] == self.max_x_.shape[0], "Number of input columns does not match scaler."
            x_col_order = np.arange(x.shape[-1])
        else:
            x_col_order = self.get_column_order(x_in_cols)
        x_transformed = x * (self.max_x_[x_col_order] - self.min_x_[x_col_order]) + self.min_x_[x_col_order]
        x_transformed = self.package_transformed_x(x_transformed, x)
        return x_transformed

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def get_scales(self):
        return self.min_x_, self.max_x_

    def __add__(self, other):
        assert type(other) is DMinMaxScaler, "Input is not DMinMaxScaler"
        assert np.all(other.x_columns_ == self.x_columns_), "Scaler columns do not match."
        current = deepcopy(self)
        current.max_x_ = np.maximum(self.max_x_, other.max_x_)
        current.min_x_ = np.minimum(self.min_x_, other.min_x_)
        return current


class DQuantileTransformer(DBaseScaler):
    """
    Distributed quantile transformer that uses the T-Digest algorithm to transform the input data into approximate
    quantiles. This class stores the centroids and counts from the T-Digest algorithm and calls pytdigest to
    perform centroid fitting and transforms of data when needed. DQuantileTransformer objects can be added together
    to create a combined quantile distribution, enabling map-reduce-style distributed calculations of quantiles
    across large number of files. The same object can also be fit multiple times to compile quantiles serially
    but with lower memory usage. Scaler supports numpy arrays, pandas dataframes, and xarray DataArrays and
    will return a transformed array in the same form as the original with column or coordinate names preserved.

    Args:
        max_merged_centroids (int): Maximum number of centroids in TDigest
        distribution (str): Output distribution of transform. Options are "uniform" (default), "normal", and "logistic".
    """
    def __init__(self, max_merged_centroids=1000, distribution="uniform"):
        self.max_merged_centroids = max_merged_centroids
        self.distribution = distribution
        self.centroids_ = None
        super().__init__()
        return

    def fit(self, x, weight=None):
        self.x_columns_, self.is_array_ = self.extract_x_columns(x)
        xv = self.extract_array(x)
        if not self._fit:
            # The number of centroids may vary depending on the distribution of each input variable.
            # The extra spots at the end are padded with nans.
            self.centroids_ = np.ones((xv.shape[-1], self.max_merged_centroids, 2)) * np.nan
            for i in range(xv.shape[-1]):
                td_obj = TDigest.compute(xv[..., i].ravel(), w=weight, compression=self.max_merged_centroids)
                td_obj.force_merge()
                self.centroids_[i, :td_obj._num_merged] = td_obj.get_centroids()
        else:
            td_objs = self.to_digests()
            new_centroids = np.ones((xv.shape[-1], self.max_merged_centroids, 2)) * np.nan
            for i, td_obj in enumerate(td_objs):
                new_td_obj = TDigest.compute(xv[..., i].ravel(), w=weight, compression=self.max_merged_centroids)
                combined_td_obj = td_obj + new_td_obj
                combined_td_obj.force_merge()
                new_centroids[i, :combined_td_obj._num_merged] = combined_td_obj.get_centroids()
            self.centroids_ = new_centroids
        self._fit = True
        return

    def to_digests(self):
        """
        Converts the centroids for each variable in DQuantileTransformer to a list of TDigest objects.

        Returns:
            td_objs: list of TDigest objects
        """
        assert self._fit, "Must call fit() before calling to_digests()"
        td_objs = []
        for i in range(self.centroids_.shape[0]):
            centroid_len = np.where(np.isnan(self.centroids_[i, :, 0]))[0].min()
            td_objs.append(TDigest.of_centroids(self.centroids_[i, :centroid_len],
                                                compression=self.max_merged_centroids))
        return td_objs

    def to_centroids(self, td_objs):
        centroids = np.ones((len(td_objs), self.max_merged_centroids, 2)) * np.nan
        for i in range(len(td_objs)):
            centroids[i, :td_objs[i]._num_merged] = td_objs[i].get_centroids()
        return centroids

    def transform(self, x):
        assert self._fit, "Scaler has not been fit."
        x_in_cols, is_array = self.extract_x_columns(x)
        if is_array:
            assert x.shape[-1] == self.centroids_.shape[0], "Number of input columns does not match scaler."
            x_col_order = np.arange(x.shape[-1])
        else:
            x_col_order = self.get_column_order(x_in_cols)
        td_objs = self.to_digests()
        xv = self.extract_array(x)
        x_transformed = np.zeros(xv.shape, dtype=xv.dtype)
        for i, o in enumerate(x_col_order):
            x_transformed[..., i] = np.reshape(td_objs[o].cdf(xv[..., i].ravel()), xv[..., i].shape)
        if self.distribution == "normal":
            x_transformed = norm.ppf(x_transformed)
        elif self.distribution == "logistic":
            x_transformed = logistic.ppf(x_transformed)
        x_transformed = self.package_transformed_x(x_transformed, x)
        return x_transformed

    def fit_transform(self, x, weight=None):
        self.fit(x, weight=weight)
        return self.transform(x)

    def inverse_transform(self, x):
        assert self._fit, "Scaler has not been fit."
        x_in_cols, is_array = self.extract_x_columns(x)
        if is_array:
            assert x.shape[-1] == self.centroids_.shape[0], "Number of input columns does not match scaler."
            x_col_order = np.arange(x.shape[-1])
        else:
            x_col_order = self.get_column_order(x_in_cols)
        xv = self.extract_array(x)
        x_transformed = np.zeros(x.shape, dtype=xv.dtype)
        td_objs = self.to_digests()
        for i, o in enumerate(x_col_order):
            x_transformed[..., i] = np.reshape(td_objs[o].inverse_cdf(xv[..., i].ravel()), xv[..., i].shape)
        if self.distribution == "normal":
            x_transformed = norm.cdf(x_transformed)
        elif self.distribution == "logistic":
            x_transformed = logistic.cdf(x_transformed)
        x_transformed = self.package_transformed_x(x_transformed, x)
        return x_transformed

    def __add__(self, other):
        assert type(other) is DQuantileTransformer, "Adding mismatched scaler types."
        assert self.is_fit() and other.is_fit(), "At least one scaler is not fit."
        td_objs = self.to_digests()
        other_td_objs = other.to_digests()
        assert len(td_objs) == len(other_td_objs), "Number of variables in scalers do not match."
        max_centroids = np.maximum(self.max_merged_centroids, other.max_merged_centroids)
        combined_centroids = np.ones((self.centroids_.shape[0], max_centroids, self.centroids_.shape[2])) * np.nan
        for i in range(len(td_objs)):
            combined_td_obj = td_objs[i] + other_td_objs[i]
            combined_td_obj.force_merge()
            combined_centroids[i, :combined_td_obj._num_merged] = combined_td_obj.get_centroids()
        new_dquantile = deepcopy(self)
        new_dquantile.max_merged_centroids = max_centroids
        new_dquantile.centroids_ = combined_centroids
        return new_dquantile
