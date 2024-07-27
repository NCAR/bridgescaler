import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured, unstructured_to_structured
from copy import deepcopy
from crick import TDigest as CTDigest
from scipy.special import ndtr, ndtri
import pandas as pd
import xarray as xr
from functools import partial
from scipy.stats import logistic
from numba import guvectorize, float32, float64, void
CENTROID_DTYPE = np.dtype([('mean', np.float64), ('weight', np.float64)])


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
            x_packaged = pd.DataFrame(x_transformed, index=x.index, columns=x.columns)
        elif hasattr(x, "coords"):
            x_packaged = xr.DataArray(x_transformed, coords=x.coords, dims=x.dims, attrs=x.attrs, name=x.name)
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
            if len(xv.shape) > 2:
                if self.channels_last:
                    self.n_ += np.prod(xv.shape[:-1])
                else:
                    self.n_ += xv.shape[0] * np.prod(xv.shape[2:])
            else:
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
                if self.channels_last:
                    x_col_order = np.arange(x.shape[-1])
                else:
                    x_col_order = np.arange(x.shape[1])
            else:
                x_col_order = self.get_column_order(x_columns)
            # update derived from
            # https://math.stackexchange.com/questions/2971315/how-do-i-combine-standard-deviations-of-two-groups
            if len(xv.shape) > 2:
                if self.channels_last:
                    new_n = np.prod(xv.shape[:-1])
                else:
                    new_n = xv.shape[0] * np.prod(xv.shape[2:])
            else:
                new_n = xv.shape[0]
            for i, o in enumerate(x_col_order):
                if self.channels_last:
                    new_mean = np.nanmean(xv[..., i])
                    new_var = np.nanvar(xv[..., i])
                else:
                    new_mean = np.nanmean(xv[:, i])
                    new_var = np.nanvar(xv[:, i])
                combined_mean = (self.n_ * self.mean_x_[o] + new_n * new_mean) / (self.n_ + new_n)
                weighted_var = (self.n_ * self.var_x_[o] + new_n * new_var) / (self.n_ + new_n)
                var_correction = (self.n_ * new_n * (self.mean_x_[o] - new_mean) ** 2) / (
                        (self.n_ + new_n) ** 2)
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
        combined_var = (self.n_ * self.var_x_ + other.n_ * other.var_x_) / (self.n_ + other.n_)
        combined_var_corr = (self.n_ * other.n_ * (self.mean_x_ - other.mean_x_) ** 2) / (
            (self.n_ + other.n_) ** 2)
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
                if self.channels_last:
                    x_col_order = np.arange(x.shape[-1])
                else:
                    x_col_order = np.arange(x.shape[1])
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


def fit_variable(var_index, xv_shared=None, compression=None, channels_last=None):
    xv = xv_shared
    td_obj = CTDigest(compression=compression)
    if channels_last:
        td_obj.update(xv[..., var_index].ravel())
    else:
        td_obj.update(xv[:, var_index].ravel())
    return td_obj


def transform_variable(td_obj, xv,
                       min_val=0.000001, max_val=0.9999999, distribution="normal"):
    td_centroids = td_obj.centroids()
    x_transformed = np.zeros_like(xv)
    tdigest_cdf(xv, td_centroids["mean"], td_centroids["weight"],
                                td_obj.min(), td_obj.max(), x_transformed)
    x_transformed = np.minimum(x_transformed, max_val)
    x_transformed = np.maximum(x_transformed, min_val)
    if distribution == "normal":
        x_transformed = ndtri(x_transformed)
    elif distribution == "logistic":
        x_transformed = logistic.ppf(x_transformed)
    return x_transformed


def inv_transform_variable(td_obj, xv,
                           distribution="normal"):
    td_centroids = td_obj.centroids()
    x_transformed = np.zeros_like(xv)
    if distribution == "normal":
        x_transformed = ndtr(xv)
    elif distribution == "logistic":
        x_transformed = logistic.cdf(xv)
    tdigest_quantile(xv, td_centroids["mean"], td_centroids["weight"],
                                td_obj.min(), td_obj.max(), x_transformed)
    return x_transformed


@guvectorize([void(float64[:], float64[:], float64[:], float64, float64, float64[:]),
            void(float32[:], float64[:], float64[:], float64, float64, float32[:])], "(m),(n),(n),(),()->(m)")
def tdigest_cdf(xv, cent_mean, cent_weight, t_min, t_max, out):
    cent_merged_weight = np.zeros_like(cent_weight)
    cumulative_weight = 0
    for i in range(cent_weight.size):
        cent_merged_weight[i] = cumulative_weight + cent_weight[i] / 2.0
        cumulative_weight += cent_weight[i]
    total_weight = cent_weight.sum()
    for i, x in enumerate(xv):
        if cent_mean.size == 0:
            out[i] = np.nan
            continue
        # Single centroid
        if cent_mean.size == 1:
            if x < t_min:
                out[i] = 0.0
            elif x > t_max:
                out[i] = 1.0
            elif t_max - t_min < np.finfo(np.float64).eps:
                out[i] = 0.5
            else:
                out[i] = (x - t_min) / (t_max - t_min)
            continue
        # Equality checks only apply if > 1 centroid
        if x >= t_max:
            out[i] = 1.0
            continue
        elif x <= t_min:
            out[i] = 0.0
            continue

        # i_l = bisect_left_mean(T->merge_centroids, x, 0, T->ncentroids);
        i_l = np.searchsorted(cent_mean, x, side="left")
        if x < cent_mean[0]:
            # min < x < first centroid
            x0 = t_min
            x1 = cent_mean[0]
            dw = cent_merged_weight[0] / 2.0
            out[i] = dw * (x - x0) / (x1 - x0) / total_weight
        elif i_l == cent_mean.size:
            # last centroid < x < max
            x0 = cent_mean[i_l - 1]
            x1 = t_max
            dw = cent_weight[i_l - 1] / 2.0
            out[i] = 1.0 - dw * (x1 - x) / (x1 - x0) / total_weight
        elif cent_mean[i_l] == x:
            # x is equal to one or more centroids
            i_r = np.searchsorted(cent_mean, x, side="right")
            out[i] = cent_merged_weight[i_r] / total_weight
        else:
            assert cent_mean[i_l] > x
            x0 = cent_mean[i_l - 1]
            x1 = cent_mean[i_l]
            dw = 0.5 * (cent_weight[i_l - 1] + cent_weight[i_l])
            out[i] = (cent_merged_weight[i_l - 1] + dw * (x - x0) / (x1 - x0)) / total_weight


@guvectorize([void(float64[:], float64[:], float64[:], float64, float64, float64[:]),
            void(float32[:], float64[:], float64[:], float64, float64, float32[:])], "(m),(n),(n),(),()->(m)")
def tdigest_quantile(qv, cent_mean, cent_weight, t_min, t_max, out):
    cent_merged_weight = np.zeros_like(cent_weight)
    cumulative_weight = 0
    for i in range(cent_weight.size):
        cent_merged_weight[i] = cumulative_weight + cent_weight[i] / 2.0
        cumulative_weight += cent_weight[i]
    total_weight = cent_weight.sum()
    for i, q in enumerate(qv):
        if total_weight == 0:
            out[i] = np.nan
            continue
        if q <= 0:
            out[i] = t_min
            continue
        if q >= 1:
            out[i] = t_max
            continue
        if cent_mean.size == 1:
            out[i] = cent_mean[0]
            continue

        index = q * total_weight
        b = np.searchsorted(cent_merged_weight, index, side="left")
        if b == 0:
            x0 = 0
            y0 = t_min
        else:
            x0 = cent_merged_weight[b - 1]
            y0 = cent_mean[b - 1]

        if b == cent_mean.size:
            x1 = total_weight
            y1 = t_max
        else:
            x1 = cent_merged_weight[b]
            y1 = cent_mean[b]
        out[i] = y0 + (index - x0) * (y1 - y0) / (x1 - x0)


class DQuantileScaler(DBaseScaler):
    """
    Distributed Quantile Scaler that uses the crick TDigest Cython library to compute quantiles across multiple
    datasets in parallel. The library can perform fitting, transforms, and inverse transforms across variables
    in parallel using the multiprocessing library. Multidimensional arrays are stored in shared memory across
    processes to minimize inter-process communication.

    Attributes:
        compression: Recommended number of centroids to use.
        distribution: "uniform", "normal", or "logistic".
        min_val: Minimum value for quantile to prevent -inf results when distribution is normal or logistic.
        max_val: Maximum value for quantile to prevent inf results when distribution is normal or logistic.
        channels_last: Whether to assume the last dim or second dim are the channel/variable dimension.
    """
    def __init__(self, compression=250, distribution="uniform", min_val=0.0000001, max_val=0.9999999, channels_last=True):
        self.compression = compression
        self.distribution = distribution
        self.min_val = min_val
        self.max_val = max_val
        self.centroids_ = None
        self.size_ = None
        self.min_ = None
        self.max_ = None

        super().__init__(channels_last=channels_last)

    def td_objs_to_attributes(self, td_objs):
        self.centroids_ = [structured_to_unstructured(td_obj.centroids()) for td_obj in td_objs]
        self.size_ = np.array([td_obj.size() for td_obj in td_objs])
        self.min_ = np.array([td_obj.min() for td_obj in td_objs])
        self.max_ = np.array([td_obj.max() for td_obj in td_objs])
        return

    def attributes_to_td_objs(self):
        td_objs = []
        if self.is_fit():
            for i in range(self.max_.size):
                td_objs.append(CTDigest(self.compression))
                td_objs[-1].__setstate__((unstructured_to_structured(self.centroids_[i], CENTROID_DTYPE),
                                          self.size_[i],
                                          self.min_[i],
                                          self.max_[i]))
        return td_objs

    def fit(self, x, weight=None):
        x_columns, is_array = self.extract_x_columns(x, channels_last=self.channels_last)
        xv = self.extract_array(x)
        channel_dim = self.set_channel_dim()
        if not self._fit:
            self.x_columns_ = x_columns
            self.is_array_ = is_array
            fit_var_func = partial(fit_variable,
                                   xv_shared=xv,
                                   compression=self.compression,
                                   channels_last=self.channels_last)
            td_objs = [fit_var_func(x) for x in np.arange(xv.shape[channel_dim])]
            self.td_objs_to_attributes(td_objs)
        else:
            assert x.shape[channel_dim] == self.x_columns_.shape[0], "New data has a different number of columns"
            if is_array:
                if self.channels_last:
                    x_col_order = np.arange(x.shape[-1])
                else:
                    x_col_order = np.arange(x.shape[1])
            else:
                x_col_order = self.get_column_order(x_columns)
            td_objs = self.attributes_to_td_objs()
            fit_var_func = partial(fit_variable,
                                   xv_shared=xv,
                                   compression=self.compression,
                                   channels_last=self.channels_last)
            new_td_objs = [fit_var_func(x) for x in np.arange(xv.shape[channel_dim])]
            for i, o in enumerate(x_col_order):
                td_objs[o].merge(new_td_objs[i])
            self.td_objs_to_attributes(td_objs)
        self._fit = True
        return

    def transform(self, x, channels_last=None, pool=None):
        xv, x_transformed, channels_last, channel_dim, x_col_order = self.process_x_for_transform(x, channels_last)
        td_objs = self.attributes_to_td_objs()
        td_i_objs = [(i, td_objs[o]) for i, o in enumerate(x_col_order)]

        trans_var_func = partial(transform_variable,
                                 min_val=self.min_val, max_val=self.max_val,
                                 distribution=self.distribution)
        if channels_last:
            if pool is not None:
                split_indices = np.round(np.linspace(0, xv[..., 0].size, pool._processes)).astype(int)
                xt_shape = x_transformed[..., 0].shape
                outputs = []
                for td_obj in td_i_objs:
                    for s, s_ind in enumerate(split_indices[1:]):
                        outputs.append(pool.apply_async(trans_var_func, (td_obj[1],
                                                        xv[..., td_obj[0]].ravel()[split_indices[s]:s_ind])))
                    x_transformed[..., td_obj[0]] = np.reshape(np.concatenate([o.get() for o in outputs]), xt_shape)
                    del outputs[:]
            else:
                for td_obj in td_i_objs:
                    x_transformed[..., td_obj[0]] = trans_var_func(td_obj[1], xv[..., td_obj[0]])
        else:
            if pool is not None:
                split_indices = np.round(np.linspace(0, xv[..., 0].size, pool._processes)).astype(int)
                xt_shape = x_transformed[:, 0].shape
                outputs = []
                for td_obj in td_i_objs:
                    for s, s_ind in enumerate(split_indices[1:]):
                        outputs.append(pool.apply_async(trans_var_func, (td_obj[1],
                                                        xv[..., td_obj[0]].ravel()[split_indices[s]:s_ind])))
                    x_transformed[:, td_obj[0]] = np.reshape(np.concatenate([o.get() for o in outputs]), xt_shape)
                    del outputs[:]
            else:
                for td_obj in td_i_objs:
                    x_transformed[:, td_obj[0]] = trans_var_func(td_obj[1], xv[:, td_obj[0]])
        x_transformed_final = self.package_transformed_x(x_transformed, x)
        return x_transformed_final

    def fit_transform(self, x, channels_last=None, weight=None, pool=None):
        self.fit(x, weight=weight)
        return self.transform(x, channels_last=channels_last, pool=pool)

    def inverse_transform(self, x, channels_last=None, pool=None):
        xv, x_transformed, channels_last, channel_dim, x_col_order = self.process_x_for_transform(x, channels_last)
        td_objs = self.attributes_to_td_objs()
        td_i_objs = [(i, td_objs[o]) for i, o in enumerate(x_col_order)]
        inv_trans_var_func = partial(inv_transform_variable,
                                 distribution=self.distribution)
        if channels_last:
            if pool is not None:
                split_indices = np.round(np.linspace(0, xv[..., 0].size, pool._processes)).astype(int)
                xt_shape = x_transformed[..., 0].shape
                outputs = []
                for td_obj in td_i_objs:
                    for s, s_ind in enumerate(split_indices[1:]):
                        outputs.append(pool.apply_async(inv_trans_var_func, (td_obj[1],
                                                        xv[..., td_obj[0]].ravel()[split_indices[s]:s_ind])))
                    x_transformed[..., td_obj[0]] = np.reshape(np.concatenate([o.get() for o in outputs]), xt_shape)
                    del outputs[:]
            else:
                for td_obj in td_i_objs:
                    x_transformed[..., td_obj[0]] = inv_trans_var_func(td_obj[1], xv[:, td_obj[0]])
        else:
            if pool is not None:
                split_indices = np.round(np.linspace(0, xv[..., 0].size, pool._processes)).astype(int)
                xt_shape = x_transformed[:, 0].shape
                outputs = []
                for td_obj in td_i_objs:
                    for s, s_ind in enumerate(split_indices[1:]):
                        outputs.append(pool.apply_async(inv_trans_var_func, (td_obj[1],
                                                        xv[..., td_obj[0]].ravel()[split_indices[s]:s_ind])))
                    x_transformed[:, td_obj[0]] = np.reshape(np.concatenate([o.get() for o in outputs]), xt_shape)
                    del outputs[:]
            else:
                for td_obj in td_i_objs:
                    x_transformed[:, td_obj[0]] = inv_trans_var_func(td_obj[1], xv[:, td_obj[0]])
        x_transformed_final = self.package_transformed_x(x_transformed, x)
        return x_transformed_final

    def __add__(self, other):
        current = deepcopy(self)
        td_objs = current.attributes_to_td_objs()
        other_td_objs = other.attributes_to_td_objs()
        assert type(other) is DQuantileScaler, "Adding mismatched scaler types."
        assert current.is_fit() and other.is_fit(), "At least one scaler is not fit."
        x_col_order = current.get_column_order(other.x_columns_)
        assert x_col_order.size > 0, "No matching columns in other DQuantileScaler"
        for i, o in enumerate(x_col_order):
            td_objs[o].merge(other_td_objs[i])
        current.td_objs_to_attributes(td_objs)
        return current

