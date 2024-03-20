import numpy as np
from copy import deepcopy
from pytdigest import TDigest
from scipy.stats import norm, logistic


class DBaseScaler(object):
    """
    Base distributed scaler class. Used only to store attributes and methods shared across all distributed
    scaler subclasses.
    """
    def __init__(self, channels_last=True):
        self.x_columns_ = None
        self.is_array_ = False
        self._fit = False
        self.channels_last = channels_last

    def is_fit(self):
        return self._fit

    @staticmethod
    def extract_x_columns(x, channels_last=True):
        """
        Extract the variable names to be transformed from x depending on if x is a pandas DataFrame, an
        xarray DataArray, or a numpy array. All of these assume that the columns are in the last dimension.
        If x is an xarray DataArray, there should be a coorindate variable with the same name as the last dimension
        of the DataArray being transformed.

        Args:
            x (Union[pandas.DataFrame, xarray.DataArray, numpy.ndarray]): array of values to be transformed.
            channels_last (bool): If True, then assume the variable or channel dimension is the last dimension of the
                array. If False, then assume the variable or channel dimension is second.

        Returns:
            xv (numpy.ndarray): Array of values to be transformed.
            is_array (bool): Whether or not x was a np.ndarray.
        """
        is_array = False
        var_dim_num = -1
        if not channels_last:
            var_dim_num = 1
        if hasattr(x, "columns"):
            x_columns = x.columns.values
        elif hasattr(x, "coords"):
            var_dim = x.dims[var_dim_num]
            x_columns = x.coords[var_dim].values
        else:
            x_columns = np.arange(x.shape[var_dim_num])
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

    def set_channel_dim(self, channels_last=None):
        if channels_last is None:
            channels_last = self.channels_last
        if channels_last:
            channel_dim = -1
        else:
            channel_dim = 1
        return channel_dim

    def process_x_for_transform(self, x, channels_last=None):
        if channels_last is None:
            channels_last = self.channels_last
        channel_dim = self.set_channel_dim(channels_last)
        assert self._fit, "Scaler has not been fit."
        x_in_cols, is_array = self.extract_x_columns(x, channels_last=channels_last)
        if is_array:
            assert x.shape[channel_dim] == self.x_columns_.shape[0], "Number of input columns does not match scaler."
            x_col_order = np.arange(x.shape[channel_dim])
        else:
            x_col_order = self.get_column_order(x_in_cols)
        xv = self.extract_array(x)
        x_transformed = np.zeros(xv.shape, dtype=xv.dtype)
        return xv, x_transformed, channels_last, channel_dim, x_col_order

    def fit(self, x, weight=None):
        pass

    def transform(self, x, channels_last=None):
        pass

    def fit_transform(self, x, channels_last=None, weight=None):
        self.fit(x, weight=weight)
        return self.transform(x, channels_last=channels_last)

    def inverse_transform(self, x, channels_last=None):
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
    def __init__(self, channels_last=True):
        self.mean_x_ = None
        self.n_ = 0
        self.var_x_ = None
        super().__init__(channels_last=channels_last)

    def fit(self, x, weight=None):
        x_columns, is_array = self.extract_x_columns(x, channels_last=self.channels_last)
        xv = self.extract_array(x)
        channel_dim = self.set_channel_dim()
        if not self._fit:
            self.x_columns_ = x_columns
            self.is_array_ = is_array
            self.n_ += xv.shape[0]
            self.mean_x_ = np.zeros(xv.shape[channel_dim], dtype=xv.dtype)
            self.var_x_ = np.zeros(xv.shape[channel_dim], dtype=xv.dtype)
            if self.channels_last:
                for i in range(xv.shape[channel_dim]):
                    self.mean_x_[i] = np.nanmean(xv[..., i])
                    self.var_x_[i] = np.nanvar(xv[..., i], ddof=1)
            else:
                for i in range(xv.shape[channel_dim]):
                    self.mean_x_[i] = np.nanmean(xv[:, i])
                    self.var_x_[i] = np.nanvar(xv[:, i], ddof=1)

        else:
            assert x.shape[channel_dim] == self.x_columns_.shape[0], "New data has a different number of columns"
            if is_array:
                x_col_order = np.arange(x.shape[-1])
            else:
                x_col_order = self.get_column_order(x_columns)
            # update derived from
            # https://math.stackexchange.com/questions/2971315/how-do-i-combine-standard-deviations-of-two-groups
            for i, o in enumerate(x_col_order):
                if self.channels_last:
                    new_mean = np.nanmean(xv[..., i])
                    new_var = np.nanvar(xv[..., i], ddof=1)
                else:
                    new_mean = np.nanmean(xv[:, i])
                    new_var = np.nanvar(xv[:, i], ddof=1)
                new_n = xv.shape[0]
                combined_mean = (self.n_ * self.mean_x_[o] + x.shape[0] * new_mean) / (self.n_ + x.shape[0])
                weighted_var = ((self.n_ - 1) * self.var_x_[o] + (new_n - 1) * new_var) / (self.n_ + new_n - 1)
                var_correction = self.n_ * new_n * (self.mean_x_[o] - new_mean) ** 2 / (
                        (self.n_ + new_n) * (self.n_ + new_n - 1))
                self.mean_x_[o] = combined_mean
                self.var_x_[o] = weighted_var + var_correction
                self.n_ += new_n
        self._fit = True

    def transform(self, x, channels_last=None):
        """
        Transform the input data from its original form to standard scaled form. If your input data has a
        different dimension order than the data used to fit the scaler, use the channels_last keyword argument
        to specify whether the new data are `channels_last` (True) or `channels_first` (False).

        Args:
            x: Input data.
            channels_last: Override the default channels_last parameter of the scaler.

        Returns:
            x_transformed: Transformed data in the same shape and type as x.
        """
        xv, x_transformed, channels_last, channel_dim, x_col_order = self.process_x_for_transform(x, channels_last)
        x_mean, x_var = self.get_scales()
        if channels_last:
            for i, o in enumerate(x_col_order):
                x_transformed[..., i] = (xv[..., i] - x_mean[o]) / np.sqrt(x_var[o])
        else:
            for i, o in enumerate(x_col_order):
                x_transformed[:, i] = (xv[:, i] - x_mean[o]) / np.sqrt(x_var[o])
        x_transformed_final = self.package_transformed_x(x_transformed, x)
        return x_transformed_final

    def inverse_transform(self, x, channels_last=None):
        xv, x_transformed, channels_last, channel_dim, x_col_order = self.process_x_for_transform(x, channels_last)
        x_mean, x_var = self.get_scales()
        if channels_last:
            for i, o in enumerate(x_col_order):
                x_transformed[..., i] = xv[..., i] * np.sqrt(x_var[o]) + x_mean[o]
        else:
            for i, o in enumerate(x_col_order):
                x_transformed[:, i] = xv[:, i] * np.sqrt(x_var[o]) + x_mean[o]
        x_transformed = self.package_transformed_x(x_transformed, x)
        return x_transformed

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
    def __init__(self, channels_last=True):
        self.max_x_ = None
        self.min_x_ = None
        super().__init__(channels_last=channels_last)

    def fit(self, x, weight=None):
        x_columns, is_array = self.extract_x_columns(x, channels_last=self.channels_last)
        xv = self.extract_array(x)
        channel_dim = self.set_channel_dim()
        if not self._fit:
            self.x_columns_ = x_columns
            self.is_array_ = is_array
            self.max_x_ = np.zeros(xv.shape[channel_dim])
            self.min_x_ = np.zeros(xv.shape[channel_dim])
            if self.channels_last:
                for i in range(xv.shape[channel_dim]):
                    self.max_x_[i] = np.nanmax(xv[..., i])
                    self.min_x_[i] = np.nanmin(xv[..., i])
            else:
                for i in range(xv.shape[channel_dim]):
                    self.max_x_[i] = np.nanmax(xv[:, i])
                    self.min_x_[i] = np.nanmin(xv[:, i])
        else:
            assert x.shape[channel_dim] == self.x_columns_.shape[0], "New data has a different number of columns"
            if is_array:
                x_col_order = np.arange(x.shape[-1])
            else:
                x_col_order = self.get_column_order(x_columns)
            if self.channels_last:
                for i, o in enumerate(x_col_order):
                    self.max_x_[o] = np.maximum(self.max_x_[o], np.nanmax(xv[..., i]))
                    self.min_x_[o] = np.minimum(self.min_x_[o], np.nanmin(xv[..., i]))
            else:
                for i, o in enumerate(xv.shape[channel_dim]):
                    self.max_x_[o] = np.maximum(self.max_x_[o], np.nanmax(xv[:, i]))
                    self.min_x_[o] = np.minimum(self.min_x_[o], np.nanmin(xv[:, i]))
        self._fit = True

    def transform(self, x, channels_last=None):
        xv, x_transformed, channels_last, channel_dim, x_col_order = self.process_x_for_transform(x, channels_last)
        if channels_last:
            for i, o in enumerate(x_col_order):
                x_transformed[..., i] = (xv[..., i] - self.min_x_[o]) / (
                        self.max_x_[o] - self.min_x_[o])
        else:
            for i, o in enumerate(x_col_order):
                x_transformed[:, i] = (xv[:, i] - self.min_x_[o]) / (
                        self.max_x_[o] - self.min_x_[o])
        x_transformed = self.package_transformed_x(x_transformed, x)
        return x_transformed

    def inverse_transform(self, x, channels_last=None):
        xv, x_transformed, channels_last, channel_dim, x_col_order = self.process_x_for_transform(x, channels_last)
        if channels_last:
            for i, o in enumerate(x_col_order):
                x_transformed[..., i] = xv[..., i] * (self.max_x_[o] -
                                                      self.min_x_[o]) + self.min_x_[o]
        else:
            for i, o in enumerate(x_col_order):
                x_transformed[:, i] = xv[:, i] * (self.max_x_[o] -
                                                  self.min_x_[o]) + self.min_x_[o]
        x_transformed = self.package_transformed_x(x_transformed, x)
        return x_transformed

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
        channels_last (bool): Whether data will use the last dimension (True) as the channels or variable dim
            or the second dimension (False).
    """
    def __init__(self, max_merged_centroids=1000, distribution="uniform", channels_last=True):
        self.max_merged_centroids = max_merged_centroids
        self.distribution = distribution
        self.centroids_ = None
        super().__init__(channels_last=channels_last)
        return

    def fit(self, x, weight=None):
        self.x_columns_, self.is_array_ = self.extract_x_columns(x, channels_last=self.channels_last)
        xv = self.extract_array(x)
        if not self._fit:
            # The number of centroids may vary depending on the distribution of each input variable.
            # The extra spots at the end are padded with nans.
            self.centroids_ = np.ones((xv.shape[-1], self.max_merged_centroids, 2)) * np.nan
            for i in range(xv.shape[-1]):
                td_obj = TDigest.compute(xv[..., i].ravel(), w=weight, compression=self.max_merged_centroids)
                self.centroids_[i, :td_obj._num_merged] = td_obj.get_centroids()
        else:
            td_objs = self.to_digests()
            new_centroids = np.ones((xv.shape[-1], self.max_merged_centroids, 2)) * np.nan
            for i, td_obj in enumerate(td_objs):
                new_td_obj = TDigest.compute(xv[..., i].ravel(), w=weight, compression=self.max_merged_centroids)
                combined_td_obj = td_obj + new_td_obj
                new_centroids[i, :combined_td_obj._num_merged] = combined_td_obj.get_centroids()
            self.centroids_ = new_centroids
        self._fit = True
        return

    def force_merge(self):
        """
        Trigger a merger of centroids for each variable.
        """
        td_objs = self.to_digests()
        for td_obj in td_objs:
            td_obj.force_merge()
        self.to_centroids(td_objs)

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

    def transform(self, x, channels_last=None):
        xv, x_transformed, channels_last, channel_dim, x_col_order = self.process_x_for_transform(x, channels_last)
        td_objs = self.to_digests()
        if channels_last:
            for i, o in enumerate(x_col_order):
                x_transformed[..., i] = np.reshape(td_objs[o].cdf(xv[..., i].ravel()), xv[..., i].shape)
        else:
            for i, o in enumerate(x_col_order):
                x_transformed[:, i] = np.reshape(td_objs[o].cdf(xv[:, i].ravel()), xv[:, i].shape)
        if self.distribution == "normal":
            x_transformed = norm.ppf(x_transformed)
        elif self.distribution == "logistic":
            x_transformed = logistic.ppf(x_transformed)
        x_transformed = self.package_transformed_x(x_transformed, x)
        return x_transformed

    def inverse_transform(self, x, channels_last=None):
        xv, x_transformed, channels_last, channel_dim, x_col_order = self.process_x_for_transform(x, channels_last)
        td_objs = self.to_digests()
        if self.distribution == "normal":
            x_transformed = norm.cdf(xv)
        elif self.distribution == "logistic":
            x_transformed = logistic.cdf(xv)
        if channels_last:
            for i, o in enumerate(x_col_order):
                x_transformed[..., i] = np.reshape(td_objs[o].inverse_cdf(x_transformed[..., i].ravel()),
                                                   xv[..., i].shape)
        else:
            for i, o in enumerate(x_col_order):
                x_transformed[:, i] = np.reshape(td_objs[o].inverse_cdf(x_transformed[:, i].ravel()),
                                                xv[:, i].shape)
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
