import warnings

from . import require_torch
require_torch()   # enforce torch availability/version at import time
import torch

from copy import deepcopy
from functools import partial

from crick import TDigest as CTDigest
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured, unstructured_to_structured
CENTROID_DTYPE = np.dtype([('mean', np.float64), ('weight', np.float64)])

warnings.simplefilter("always")


class DBaseScalerTensor:
    """
    Base distributed scaler class for torch.Tensor. Used only to store attributes and methods
    shared across all distributed scaler subclasses.
    """

    def __init__(self, channels_last=True):
        self.x_columns_ = None
        self._fit = False
        self.channels_last = channels_last

    def is_fit(self):
        return self._fit

    def extract_x_columns(self, x, channels_last=True):
        """
        Extract the variable names from input x.

        The variable names are expected to be stored in the `variable_names`
        attribute of the torch.Tensor. If the attribute is missing, a warning is
        issued to notify the user that alignment validation will be limited.

        Args:
            x (torch.Tensor): The input tensor containing data and optionally the
                `variable_names` attribute.
            channels_last (bool): If True, then assume the variable or channel dimension
                is the last dimension of the array. If False, then assume the variable or channel
                dimension is second.

        Returns:
            x_columns (list[str] | list[int]): Variable names if available; otherwise,
                integer indices generated based on the length of the variable/channel dimension.

        Raises:
            TypeError: If `x` is not a torch.Tensor or if `variable_names`
                is not a list.
            ValueError: If `variable_names` contains duplicate entries.
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Input must be a PyTorch tensor, not {type(x).__name__}.")

        # Access attribute
        x_columns, has_attribute = getattr(x, 'variable_names', None), hasattr(x, 'variable_names')

        # 1. Missing attribute
        if x_columns is None:
            warnings.warn(
                f"Input data lacks variable names. When performing fit or transform, "
                f"data/scaler consistency check is limited to variable counts; "
                f"order cannot be validated. Ensure variable alignment to prevent incorrect results."
            )
            x_columns = list(range(x.shape[self.set_channel_dim(channels_last)]))
            return x_columns, has_attribute

        # 2. Attribute exists and type check
        if not isinstance(x_columns, list):
            raise TypeError(
                f"Attribute variable_names must be a list, but received {type(x_columns).__name__}. "
                f"Please provide a list of strings (e.g., ['var1', 'var2'])."
            )

        # 3. Check for data and attribute "variable_names" consistency
        channel_dim = self.set_channel_dim()
        data_channels = x.shape[channel_dim]
        attribute_channels = len(x_columns)

        if attribute_channels != data_channels:
            raise ValueError(
                f"Input data channel dimension mismatch: "
                f"data has {data_channels} channels (dim={channel_dim}), "
                f"but {attribute_channels} were found in attribute variable_names."
            )

        # 4. Check for duplicates
        if len(set(x_columns)) != len(x_columns):
            raise ValueError(
                f"Duplicates found in variable_names! "
                f"{len(set(x_columns))} unique vs {len(x_columns)} total."
            )

        return x_columns, has_attribute

    @staticmethod
    def extract_array(x):
        pass

    def get_column_order(self, x_in_columns):
        """
        Get the indices of the scaler columns that have the same name as the variables (columns) in the input x tensor. This
        enables users to pass a torch.Tensor to transform or inverse_transform with fewer variables than
        the original scaler or variables in a different order and still have the input dataset be transformed properly.

        Args:
            x_in_columns (list): list of input variable names.

        Returns:
            x_in_col_indices (list): integer indices of the input variables from x in the scaler in order.
        """
        assert all(var in self.x_columns_ for var in x_in_columns), (
            f"Some input variables not in scaler x_columns. "
            f"Scaler: {self.x_columns_}, input variables: {x_in_columns}"
        )
        x_in_col_indices = [self.x_columns_.index(item) for item in x_in_columns if item in self.x_columns_]
        return x_in_col_indices

    @staticmethod
    def package_transformed_x(x_transformed, x):
        """
        Repackaged a transformed torch.Tensor into the same datatype as the original x, including
        all metadata.

        Args:
            x_transformed (torch.Tensor): array after being transformed or inverse transformed
            x (torch.Tensor): original data

        Returns:

        """
        x_packaged = x_transformed
        if getattr(x, 'variable_names', None) is not None:
            x_packaged.variable_names = x.variable_names
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
        x_in_cols, has_attribute = self.extract_x_columns(x, channels_last=channels_last)
        if not has_attribute:
            assert (
                    x.shape[channel_dim] == len(self.x_columns_)
            ), "Number of input variables does not match scaler."
        x_col_order = self.get_column_order(x_in_cols)
        xv = x
        x_transformed = torch.zeros(xv.shape, dtype=xv.dtype, device=xv.device)
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

    @staticmethod
    def reshape_to_channels_first(stat, target):
        """Reshapes 'stat' to align with the channel dimension (index 1)."""
        return stat.view(*(stat.size(0) if i == 1 else 1 for i in range(target.dim())))

    @staticmethod
    def reshape_to_channels_last(stat, target):
        """Reshapes 'stat' to align with the last dimension."""
        return stat.view(*(stat.size(0) if i == target.dim() - 1 else 1 for i in range(target.dim())))


class DStandardScalerTensor(DBaseScalerTensor):
    """
    Distributed version of StandardScaler. You can calculate this map-reduce style by running it on individual
    data files, returning the fitted objects, and then summing them together to represent the full dataset. Scaler
    supports torch.Tensor and returns a transformed tensor.
    """

    def __init__(self, channels_last=True):
        self.mean_x_ = None
        self.n_ = 0
        self.var_x_ = None
        super().__init__(channels_last=channels_last)

    def fit(self, x, weight=None):
        x_columns, has_attribute = self.extract_x_columns(x, channels_last=self.channels_last)
        xv = x
        channel_dim = self.set_channel_dim()
        if not self._fit:
            self.x_columns_ = x_columns
            if len(xv.shape) > 2:
                if self.channels_last:
                    self.n_ += torch.prod(torch.tensor(xv.shape[:-1], dtype=xv.dtype, device=xv.device))
                else:
                    self.n_ += xv.shape[0] * \
                        torch.prod(torch.tensor(xv.shape[2:], dtype=xv.dtype, device=xv.device))
            else:
                self.n_ += xv.shape[0]
            self.mean_x_ = torch.zeros(xv.shape[channel_dim], dtype=xv.dtype, device=xv.device)
            self.var_x_ = torch.zeros(xv.shape[channel_dim], dtype=xv.dtype, device=xv.device)

            if self.channels_last:
                self.mean_x_ = torch.mean(xv, dim=tuple(range(xv.ndim - 1)))
                self.var_x_ = torch.var(xv, dim=tuple(range(xv.ndim - 1)), correction=0)
            else:
                self.mean_x_ = torch.mean(xv, dim=tuple(d for d in range(xv.ndim) if d != 1))
                self.var_x_ = torch.var(xv, dim=tuple(d for d in range(xv.ndim) if d != 1), correction=0)

        else:
            # Update existing scaler with new data
            assert (
                    x.shape[channel_dim] == len(self.x_columns_)
            ), "New data has a different number of variables."
            x_col_order = self.get_column_order(x_columns)
            if len(xv.shape) > 2:
                if self.channels_last:
                    new_n = torch.prod(torch.tensor(xv.shape[:-1], dtype=xv.dtype, device=xv.device))
                else:
                    new_n = xv.shape[0] * \
                        torch.prod(torch.tensor(xv.shape[2:], dtype=xv.dtype, device=xv.device))
            else:
                new_n = xv.shape[0]
            if self.channels_last:
                new_mean = torch.mean(xv[...,x_col_order], dim=tuple(range(xv.ndim - 1)))
                new_var = torch.var(xv[...,x_col_order], dim=tuple(range(xv.ndim - 1)), correction=0)
            else:
                new_mean = torch.mean(xv[:, x_col_order], dim=tuple(d for d in range(xv.ndim) if d != 1))
                new_var = torch.var(xv[:, x_col_order], dim=tuple(d for d in range(xv.ndim) if d != 1), correction=0)
            combined_mean = (self.n_ * self.mean_x_ + new_n * new_mean) / (
                self.n_ + new_n
            )
            weighted_var = (self.n_ * self.var_x_ + new_n * new_var) / (
                self.n_ + new_n
            )
            var_correction = (
                self.n_ * new_n * (self.mean_x_ - new_mean) ** 2
            ) / ((self.n_ + new_n) ** 2)
            self.mean_x_ = combined_mean
            self.var_x_ = weighted_var + var_correction
            self.n_ += new_n
        self._fit = True

    def transform(self, x, channels_last=None):
        """
        Transform the input data from its original form to standard scaled form. If your input data has a
        different dimension order than the data used to fit the scaler, use the channels_last keyword argument
        to specify whether the new data are `channels_last` (True) or `channels_first` (False).

        Args:
            x (torch.Tensor): Input data.
            channels_last: Override the default channels_last parameter of the scaler.

        Returns:
            x_transformed (torch.Tensor): Transformed data in the same shape and type as x.
        """
        (
            xv,
            x_transformed,
            channels_last,
            channel_dim,
            x_col_order,
        ) = self.process_x_for_transform(x, channels_last)
        x_mean, x_var = self.get_scales(x_col_order)
        if channels_last:
            x_transformed = (
                    xv - self.reshape_to_channels_last(x_mean.to(device=xv.device), xv)) / torch.sqrt(self.reshape_to_channels_last(x_var.to(device=xv.device), xv))
        else:
            x_transformed = (
                    xv - self.reshape_to_channels_first(x_mean.to(device=xv.device), xv)) / torch.sqrt(self.reshape_to_channels_first(x_var.to(device=xv.device), xv))
        x_transformed_final = self.package_transformed_x(x_transformed, x)
        return x_transformed_final

    def inverse_transform(self, x, channels_last=None):
        (
            xv,
            x_transformed,
            channels_last,
            channel_dim,
            x_col_order,
        ) = self.process_x_for_transform(x, channels_last)
        x_mean, x_var = self.get_scales(x_col_order)
        if channels_last:
            x_transformed = xv * \
                    torch.sqrt(self.reshape_to_channels_last(x_var.to(device=xv.device), xv)) + self.reshape_to_channels_last(x_mean.to(device=xv.device), xv)
        else:
            x_transformed = xv * \
                    torch.sqrt(self.reshape_to_channels_first(x_var.to(device=xv.device), xv)) + self.reshape_to_channels_first(x_mean.to(device=xv.device), xv)
        x_transformed_final = self.package_transformed_x(x_transformed, x)
        return x_transformed_final

    def get_scales(self, x_col_order=slice(None)):
        return self.mean_x_[x_col_order], self.var_x_[x_col_order]

    def __add__(self, other):
        current = deepcopy(self)

        assert type(other) is DStandardScalerTensor, "Input is not DStandardScalerTensor."
        assert current.is_fit() and other.is_fit(), "At least one scaler is not fit."
        assert other.x_columns_ == self.x_columns_, "Scaler variables do not match."

        current.mean_x_ = (self.n_ * self.mean_x_ + other.n_ * other.mean_x_) / (self.n_ + other.n_)
        combined_var = (self.n_ * self.var_x_ + other.n_ * other.var_x_) / (self.n_ + other.n_)
        combined_var_corr = (self.n_ * other.n_ * (self.mean_x_ - other.mean_x_) ** 2) / ((self.n_ + other.n_) ** 2)
        current.var_x_ = combined_var + combined_var_corr
        current.n_ = self.n_ + other.n_
        return current


class DMinMaxScalerTensor(DBaseScalerTensor):
    """
    Distributed MinMaxScaler enables calculation of min and max of variables in datasets in parallel, then combining
    the mins and maxes as a reduction step. Scaler supports torch.Tensor and will return a transformed tensor in the
    same form as the original with variable/column names preserved.
    """

    def __init__(self, channels_last=True):
        self.max_x_ = None
        self.min_x_ = None
        super().__init__(channels_last=channels_last)

    def fit(self, x, weight=None):
        x_columns, has_attribute = self.extract_x_columns(x, channels_last=self.channels_last)
        xv = x
        channel_dim = self.set_channel_dim()
        if not self._fit:
            self.x_columns_ = x_columns
            self.max_x_ = torch.zeros(xv.shape[channel_dim], dtype=xv.dtype, device=xv.device)
            self.min_x_ = torch.zeros(xv.shape[channel_dim], dtype=xv.dtype, device=xv.device)

            if self.channels_last:
                self.max_x_ = torch.amax(xv, dim=tuple(range(xv.ndim - 1)))
                self.min_x_ = torch.amin(xv, dim=tuple(range(xv.ndim - 1)))
            else:
                self.max_x_ = torch.amax(xv, dim=tuple(d for d in range(xv.ndim) if d != 1))
                self.min_x_ = torch.amin(xv, dim=tuple(d for d in range(xv.ndim) if d != 1))
        else:
            # Update existing scaler with new data
            assert (
                    x.shape[channel_dim] == len(self.x_columns_)
            ), "New data has a different number of variables."
            x_col_order = self.get_column_order(x_columns)
            if self.channels_last:
                self.max_x_ = torch.maximum(
                    self.max_x_, torch.amax(xv[...,x_col_order], dim=tuple(range(xv.ndim - 1)))
                )
                self.min_x_ = torch.minimum(
                    self.min_x_, torch.amin(xv[...,x_col_order], dim=tuple(range(xv.ndim - 1)))
                )
            else:
                self.max_x_ = torch.maximum(
                    self.max_x_, torch.amax(xv[:, x_col_order], dim=tuple(d for d in range(xv.ndim) if d != 1)))
                self.min_x_ = torch.minimum(
                    self.min_x_, torch.amin(xv[:, x_col_order], dim=tuple(d for d in range(xv.ndim) if d != 1)))
        self._fit = True

    def transform(self, x, channels_last=None):
        (
            xv,
            x_transformed,
            channels_last,
            channel_dim,
            x_col_order,
        ) = self.process_x_for_transform(x, channels_last)
        x_min, x_max = self.get_scales(x_col_order)
        if channels_last:
            x_transformed = (xv - self.reshape_to_channels_last(x_min.to(device=xv.device), xv)) / (
                self.reshape_to_channels_last(x_max.to(device=xv.device), xv) - self.reshape_to_channels_last(x_min.to(device=xv.device), xv)
            )
        else:
            x_transformed = (xv - self.reshape_to_channels_first(x_min.to(device=xv.device), xv)) / (
                self.reshape_to_channels_first(x_max.to(device=xv.device), xv) - self.reshape_to_channels_first(x_min.to(device=xv.device), xv)
            )
        x_transformed_final = self.package_transformed_x(x_transformed, x)
        return x_transformed_final

    def inverse_transform(self, x, channels_last=None):
        (
            xv,
            x_transformed,
            channels_last,
            channel_dim,
            x_col_order,
        ) = self.process_x_for_transform(x, channels_last)
        x_min, x_max = self.get_scales(x_col_order)
        if channels_last:
            x_transformed = (
                xv * (self.reshape_to_channels_last(x_max.to(device=xv.device), xv) - self.reshape_to_channels_last(x_min.to(device=xv.device), xv)
                                ) + self.reshape_to_channels_last(x_min.to(device=xv.device), xv)
            )
        else:
            x_transformed = (
                xv * (self.reshape_to_channels_first(x_max.to(device=xv.device), xv) - self.reshape_to_channels_first(x_min.to(device=xv.device), xv)) +
                self.reshape_to_channels_first(x_min.to(device=xv.device), xv)
            )
        x_transformed_final = self.package_transformed_x(x_transformed, x)
        return x_transformed_final

    def get_scales(self, x_col_order=slice(None)):
        return self.min_x_[x_col_order], self.max_x_[x_col_order]

    def __add__(self, other):
        current = deepcopy(self)

        assert type(other) is DMinMaxScalerTensor, "Input is not DMinMaxScalerTensor."
        assert current.is_fit() and other.is_fit(), "At least one scaler is not fit."
        assert other.x_columns_ == self.x_columns_, "Scaler variables do not match."

        current.max_x_ = torch.maximum(self.max_x_, other.max_x_)
        current.min_x_ = torch.minimum(self.min_x_, other.min_x_)
        return current


def fit_variable_tensor(var_index, xv, compression=None, channels_last=None):
    # Check if the tensor is on GPU
    if xv.is_cuda:
        warnings.warn(
            "Performance Warning: Tensor is on GPU. Keep data on CPU for fitting to "
            "avoid implicit GPU-to-CPU copies and performance slowdowns.",
            RuntimeWarning,
            stacklevel = 2
        )
    xv_nd = xv.cpu().numpy()  # ensure the tensor is on CPU and then convert it into ndarray.

    td_obj = CTDigest(compression=compression)
    if channels_last:
        td_obj.update(xv_nd[..., var_index].ravel())
    else:
        td_obj.update(xv_nd[:, var_index].ravel())
    return td_obj


def transform_variable_tensor(cent_mean, cent_weight, t_min, t_max, xv,
                       min_val=0.000001, max_val=0.9999999, distribution="normal"):
    x_transformed = tdigest_cdf_tensor(xv, cent_mean, cent_weight, t_min, t_max)
    x_transformed = torch.minimum(x_transformed, torch.tensor(max_val, dtype=x_transformed.dtype,
                                                              device=x_transformed.device))
    x_transformed = torch.maximum(x_transformed, torch.tensor(min_val, dtype=x_transformed.dtype,
                                                              device=x_transformed.device))
    if distribution == "normal":
        x_transformed = torch.special.ndtri(x_transformed)
    elif distribution == "logistic":
        x_transformed = torch.logit(x_transformed)
    return x_transformed


def inv_transform_variable_tensor(cent_mean, cent_weight, t_min, t_max, xv,
                                  distribution="normal"):
    if distribution == "normal":
        x_intermediate = torch.special.ndtr(xv)
    elif distribution == "logistic":
        x_intermediate = torch.sigmoid(xv)
    else:
        x_intermediate = xv
    x_transformed = tdigest_quantile_tensor(x_intermediate, cent_mean, cent_weight, t_min, t_max)
    return x_transformed


def tdigest_cdf_tensor(xv, cent_mean, cent_weight, t_min, t_max):
    num_centroids = cent_mean.numel()

    cum_sum = torch.cumsum(cent_weight, dim=0)
    cent_merged_weight = cum_sum - (cent_weight / 2.0)
    total_weight = cent_weight.sum()

    out = torch.full(xv.shape, torch.nan, dtype=xv.dtype, device=xv.device)

    if num_centroids == 0:
        return out

    # Single centroid
    if num_centroids == 1:
        out = torch.where(xv < t_min, 0.0, out)
        out = torch.where(xv > t_max, 1.0, out)

        mask = (xv >= t_min) & (xv <= t_max)
        eps = torch.finfo(xv.dtype).eps # find the Machine Epsilon
        if t_max - t_min < eps: # smaller than the smallest measurable gap
            out[mask] = 0.5
        else:
            out[mask] = (xv[mask] - t_min) / (t_max - t_min)
        return out

    # Multi-centroid
    # clamping extremes
    out = torch.where(xv >= t_max, 1.0, out)
    out = torch.where(xv <= t_min, 0.0, out)

    # identify indices that still need processing
    active_mask = torch.isnan(out)
    if not active_mask.any():
        return out

    x_active = xv[active_mask]

    # binary Search
    # i_l is the index where x_active would be inserted to maintain order, "cent_mean" is a sorted tensor
    i_l = torch.searchsorted(cent_mean, x_active, side="left") # (default): $S[i-1] < v \le S[i]$

    # initialize results for active elements
    res = torch.zeros_like(x_active)

    # --- min < x < first centroid ---
    m1 = x_active < cent_mean[0]
    if m1.any():
        x0, x1 = t_min, cent_mean[0]
        dw = cent_merged_weight[0] / 2.0
        res[m1] = dw * (x_active[m1] - x0) / (x1 - x0) / total_weight

    # --- last centroid < x < max ---
    m2 = (~m1) & (i_l == num_centroids)
    if m2.any():
        idx_last = num_centroids - 1
        x0, x1 = cent_mean[idx_last], t_max
        dw = cent_weight[idx_last] / 2.0
        res[m2] = 1.0 - dw * (x1 - x_active[m2]) / (x1 - x0) / total_weight

    # --- x is equal to one or more centroids ---
    m3 = (~m1) & (~m2) & (cent_mean[i_l.clamp(max=num_centroids - 1)] == x_active)
    if m3.any():
        # side='right' finds the upper bound of the equality range
        i_r = torch.searchsorted(cent_mean, x_active, side="right")
        res[m3] = cent_merged_weight[i_r[m3]] / total_weight

    # --- x between two centroids ---
    m4 = (~m1) & (~m2) & (~m3)
    if m4.any():
        idx_l = i_l[m4]
        x0 = cent_mean[idx_l - 1]
        x1 = cent_mean[idx_l]
        dw = 0.5 * (cent_weight[idx_l - 1] + cent_weight[idx_l])
        interpolated = cent_merged_weight[idx_l - 1] + dw * (x_active[m4] - x0) / (x1 - x0)
        res[m4] = interpolated / total_weight

    out[active_mask] = res
    return out


def tdigest_quantile_tensor(qv, cent_mean, cent_weight, t_min, t_max):
    num_centroids = cent_mean.numel()

    cum_sum = torch.cumsum(cent_weight, dim=0)
    cent_merged_weight = cum_sum - (cent_weight / 2.0)
    total_weight = cent_weight.sum()

    out = torch.full(qv.shape, torch.nan, dtype=qv.dtype, device=qv.device)

    if total_weight == 0:
        return out

    out = torch.where(qv <= 0, t_min, out)
    out = torch.where(qv >= 1, t_max, out)

    if num_centroids == 1:
        mask = (qv > 0) & (qv < 1)
        out[mask] = cent_mean[0]  # valid ones return the "mean"
        return out

    # identify indices that still need processing
    active_mask = torch.isnan(out)
    if not active_mask.any():
        return out

    q_active = qv[active_mask]
    target_weight = q_active * total_weight

    idx_r = torch.searchsorted(cent_merged_weight, target_weight, side="left")
    idx_l = idx_r - 1

    # --- define x0, y0 (Left boundary of interpolation) ---
    # If idx_r is 0, we interpolate from (0, t_min)
    x0 = torch.where(idx_r == 0, 0.0, cent_merged_weight[idx_l.clamp(min=0)])
    y0 = torch.where(idx_r == 0, t_min, cent_mean[idx_l.clamp(min=0)])

    # --- define x1, y1 (right boundary of interpolation) ---
    at_end = (idx_r == num_centroids)
    x1 = torch.where(at_end, total_weight, cent_merged_weight[idx_r.clamp(max=num_centroids-1)])
    y1 = torch.where(at_end, t_max, cent_mean[idx_r.clamp(max=num_centroids-1)])

    # --- linear interpolation formula: y = y0 + (x - x0) * (y1 - y0) / (x1 - x0) ---
    denom = x1 - x0
    res = y0 + (target_weight - x0) * (y1 - y0) / torch.where(denom == 0, 1e-9, denom)

    out[active_mask] = res
    return out


class DQuantileScalerTensor(DBaseScalerTensor):
    """
    Distributed Quantile Scaler for tensors that uses the crick TDigest Cython library to compute quantiles across multiple
    datasets in parallel. The library can perform fitting, transforms, and inverse transforms.

    DQuantileScaler supports

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
        self.centroids_mean_tensor = None
        self.centroids_weight_tensor = None
        self.min_tensor = None
        self.max_tensor = None

        super().__init__(channels_last=channels_last)

    # method copied from distributed.py
    def td_objs_to_attributes(self, td_objs):
        self.centroids_ = [structured_to_unstructured(td_obj.centroids()) for td_obj in td_objs]
        self.size_ = np.array([td_obj.size() for td_obj in td_objs])
        self.min_ = np.array([td_obj.min() for td_obj in td_objs])
        self.max_ = np.array([td_obj.max() for td_obj in td_objs])
        return

    # method copied from distributed.py
    def attributes_to_td_objs(self):
        td_objs = []
        if self.is_fit():
            for i in range(len(self.max_)):
                td_objs.append(CTDigest(self.compression))
                td_objs[-1].__setstate__((unstructured_to_structured(self.centroids_[i], CENTROID_DTYPE),
                                          self.size_[i],
                                          self.min_[i],
                                          self.max_[i]))
        return td_objs

    def tensorize_attributes(self):
        self.centroids_mean_tensor = [torch.from_numpy(c[:, 0]) for c in self.centroids_]    # "mean"
        self.centroids_weight_tensor = [torch.from_numpy(c[:, 1]) for c in self.centroids_]  # "weight"
        self.min_tensor = torch.from_numpy(self.min_)
        self.max_tensor = torch.from_numpy(self.max_)
        return

    def fit(self, x, weight=None):
        x_columns, has_attribute = self.extract_x_columns(x, channels_last=self.channels_last)
        xv = x
        channel_dim = self.set_channel_dim()
        if not self._fit:
            self.x_columns_ = x_columns
            fit_var_func = partial(fit_variable_tensor,
                                   xv=xv,
                                   compression=self.compression,
                                   channels_last=self.channels_last)
            td_objs = [fit_var_func(x) for x in range(xv.shape[channel_dim])]
            self.td_objs_to_attributes(td_objs)
            self.tensorize_attributes()
        else:
            # Update existing scaler with new data
            assert (
                    x.shape[channel_dim] == len(self.x_columns_)
            ), "New data has a different number of variables."
            x_col_order = self.get_column_order(x_columns)
            td_objs = self.attributes_to_td_objs()
            fit_var_func = partial(fit_variable_tensor,
                                   xv=xv,
                                   compression=self.compression,
                                   channels_last=self.channels_last)
            new_td_objs = [fit_var_func(x) for x in range(xv.shape[channel_dim])]
            for i, o in enumerate(x_col_order):
                td_objs[o].merge(new_td_objs[i])
            self.td_objs_to_attributes(td_objs)
            self.tensorize_attributes()
        self._fit = True
        return

    def transform(self, x, channels_last=None):
        xv, x_transformed, channels_last, channel_dim, x_col_order = self.process_x_for_transform(x, channels_last)

        trans_var_func = partial(transform_variable_tensor,
                                 min_val=self.min_val, max_val=self.max_val,
                                 distribution=self.distribution)

        if channels_last:
            for i, o in enumerate(x_col_order):
                x_transformed[..., i] = trans_var_func(self.centroids_mean_tensor[o].to(xv.device, dtype=xv.dtype),
                                                       self.centroids_weight_tensor[o].to(xv.device, dtype=xv.dtype),
                                                       self.min_tensor[o].to(xv.device, dtype=xv.dtype),
                                                       self.max_tensor[o].to(xv.device, dtype=xv.dtype), xv[..., i])
        else:
            for i, o in enumerate(x_col_order):
                x_transformed[:, i] = trans_var_func(self.centroids_mean_tensor[o].to(xv.device, dtype=xv.dtype),
                                                     self.centroids_weight_tensor[o].to(xv.device, dtype=xv.dtype),
                                                     self.min_tensor[o].to(xv.device, dtype=xv.dtype),
                                                     self.max_tensor[o].to(xv.device, dtype=xv.dtype), xv[:, i])

        x_transformed_final = self.package_transformed_x(x_transformed, x)
        return x_transformed_final

    def fit_transform(self, x, channels_last=None, weight=None):
        self.fit(x, weight=weight)
        return self.transform(x, channels_last=channels_last)

    def inverse_transform(self, x, channels_last=None):
        xv, x_transformed, channels_last, channel_dim, x_col_order = self.process_x_for_transform(x, channels_last)

        inv_trans_var_func = partial(inv_transform_variable_tensor,
                                     distribution=self.distribution)
        if channels_last:
            for i, o in enumerate(x_col_order):
                x_transformed[..., i] = inv_trans_var_func(self.centroids_mean_tensor[o].to(xv.device, dtype=xv.dtype),
                                                           self.centroids_weight_tensor[o].to(xv.device, dtype=xv.dtype),
                                                           self.min_tensor[o], self.max_tensor[o], xv[..., i])
        else:
            for i, o in enumerate(x_col_order):
                x_transformed[:, i] = inv_trans_var_func(self.centroids_mean_tensor[o].to(xv.device, dtype=xv.dtype),
                                                         self.centroids_weight_tensor[o].to(xv.device, dtype=xv.dtype),
                                                         self.min_tensor[o], self.max_tensor[o], xv[:, i])

        x_transformed_final = self.package_transformed_x(x_transformed, x)
        return x_transformed_final

    def __add__(self, other):
        current = deepcopy(self)

        assert type(other) is DQuantileScalerTensor, "Input is not DQuantileScalerTensor."
        assert current.is_fit() and other.is_fit(), "At least one scaler is not fit."
        assert other.x_columns_ == self.x_columns_, "Scaler variables do not match."

        td_objs = current.attributes_to_td_objs()
        other_td_objs = other.attributes_to_td_objs()
        for i in range(len(td_objs)):
            td_objs[i].merge(other_td_objs[i])
        current.td_objs_to_attributes(td_objs)
        current.tensorize_attributes()
        return current