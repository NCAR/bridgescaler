import warnings

from . import require_torch
require_torch()   # enforce torch availability/version at import time
import torch

import weakref
from copy import deepcopy
from functools import partial

from crick import TDigest as CTDigest
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured, unstructured_to_structured
CENTROID_DTYPE = np.dtype([('mean', np.float64), ('weight', np.float64)])

warnings.simplefilter("always")

# Cache of vmapped (and optionally torch.compiled) transform functions, keyed weakly by scaler instance so
# nothing non-serializable is stored on the scaler's __dict__ and entries are dropped when the scaler is freed.
_BATCHED_CACHE = weakref.WeakKeyDictionary()


def _bucketize(sorted_seq, values, side):
    """Return insertion indices of ``values`` into the ascending 1-D ``sorted_seq`` (like ``torch.searchsorted``).

    ``torch.searchsorted`` has a pathologically slow Metal kernel, so on MPS the indices are computed with a
    broadcast reduction instead (O(N*K) memory, but roughly an order of magnitude faster there). On CPU/CUDA the
    binary-search ``torch.searchsorted`` is used, which is far faster and memory-light where it is well
    implemented. The device check is a static branch, so it does not break the graph under ``torch.compile``.

    Args:
        sorted_seq (torch.Tensor): ascending boundaries, shape ``(K,)`` (per variable under vmap).
        values (torch.Tensor): values to locate, any shape.
        side (str): ``"left"`` (count of boundaries strictly less than each value) or ``"right"`` (count of
            boundaries less than or equal to each value), matching ``torch.searchsorted`` semantics.

    Returns:
        torch.Tensor: int64 indices with the same shape as ``values``.
    """
    if values.device.type == "mps":
        if side == "left":
            return torch.sum(sorted_seq < values.unsqueeze(-1), dim=-1)
        return torch.sum(sorted_seq <= values.unsqueeze(-1), dim=-1)
    return torch.searchsorted(sorted_seq, values, side=side)


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


def transform_variable_tensor(cent_mean, cent_weight, t_min, t_max, n_real, xv,
                       min_val=0.000001, max_val=0.9999999, distribution="normal"):
    """Per-variable forward quantile transform, written branchlessly so it can be wrapped in ``torch.vmap``.

    ``cent_mean``/``cent_weight`` are the padded ``(K,)`` centroid tensors for a single variable (padding means
    are ``+inf`` and padding weights are ``0``); ``n_real`` is the number of valid (non-padding) centroids.
    """
    x_transformed = tdigest_cdf_tensor(xv, cent_mean, cent_weight, t_min, t_max, n_real)
    x_transformed = torch.clamp(x_transformed, min=min_val, max=max_val)
    if distribution == "normal":
        x_transformed = torch.special.ndtri(x_transformed)
    elif distribution == "logistic":
        x_transformed = torch.logit(x_transformed)
    return x_transformed


def inv_transform_variable_tensor(cent_mean, cent_weight, t_min, t_max, n_real, xv,
                                  distribution="normal"):
    """Per-variable inverse quantile transform, branchless for ``torch.vmap``. See ``transform_variable_tensor``."""
    if distribution == "normal":
        x_intermediate = torch.special.ndtr(xv)
    elif distribution == "logistic":
        x_intermediate = torch.sigmoid(xv)
    else:
        x_intermediate = xv
    x_transformed = tdigest_quantile_tensor(x_intermediate, cent_mean, cent_weight, t_min, t_max, n_real)
    return x_transformed


def tdigest_cdf_tensor(xv, cent_mean, cent_weight, t_min, t_max, n_real):
    """Evaluate the TDigest CDF at every element of ``xv`` for a single variable.

    This is a branchless, fixed-shape reformulation of the original masked implementation so that it composes
    with ``torch.vmap`` over the channel dimension. All four interpolation cases are computed everywhere and
    selected with ``torch.where`` rather than boolean-mask assignment, and centroid arrays are padded to a
    common length ``K`` (``+inf`` means, ``0`` weights) so that ``n_real`` (not ``K``) delimits the real
    centroids.

    Args:
        xv (torch.Tensor): values to evaluate, any shape.
        cent_mean (torch.Tensor): padded ``(K,)`` centroid means, ascending, padding = ``+inf``.
        cent_weight (torch.Tensor): padded ``(K,)`` centroid weights, padding = ``0``.
        t_min, t_max: scalar min/max of the digest.
        n_real: number of valid centroids (0-d integer tensor works under vmap).

    Returns:
        torch.Tensor: CDF values in ``[0, 1]`` with the same shape as ``xv``.
    """
    K = cent_mean.numel()
    cum_sum = torch.cumsum(cent_weight, dim=0)
    cent_merged_weight = cum_sum - (cent_weight / 2.0)
    total_weight = cent_weight.sum()
    eps = torch.finfo(xv.dtype).eps

    # index of the last real centroid, and clamped search indices that never reach the padding region
    nlast = torch.clamp(n_real - 1, min=0)
    i_l = _bucketize(cent_mean, xv, "left")   # in [0, K]
    i_r = _bucketize(cent_mean, xv, "right")  # in [0, K]
    i_l_c = torch.minimum(i_l, nlast)
    i_r_c = torch.minimum(i_r, nlast)

    first_mean = cent_mean[0]
    last_mean = cent_mean[nlast]
    last_w = cent_weight[nlast]

    # --- case m1: t_min < x < first centroid ---
    dw1 = cent_merged_weight[0] / 2.0
    denom1 = first_mean - t_min
    res_m1 = dw1 * (xv - t_min) / torch.where(denom1 == 0, 1.0, denom1) / total_weight

    # --- case m2: last centroid < x < t_max ---
    dw2 = last_w / 2.0
    denom2 = t_max - last_mean
    res_m2 = 1.0 - dw2 * (t_max - xv) / torch.where(denom2 == 0, 1.0, denom2) / total_weight

    # --- case m3: x equals a centroid mean ---
    res_m3 = cent_merged_weight[i_r_c] / total_weight

    # --- case m4: x strictly between two centroids ---
    # keep idx_l in [1, K-1] so the (discarded) result never indexes out of bounds when K == 1
    idx_l = torch.minimum(torch.clamp(i_l, min=1), torch.clamp(nlast, min=1)).clamp(max=max(K - 1, 0))
    x0 = cent_mean[idx_l - 1]
    x1 = cent_mean[idx_l]
    dw4 = 0.5 * (cent_weight[idx_l - 1] + cent_weight[idx_l])
    denom4 = x1 - x0
    res_m4 = (cent_merged_weight[idx_l - 1] + dw4 * (xv - x0) / torch.where(denom4 == 0, 1.0, denom4)) / total_weight

    # select case by priority m1 > m2 > m3 > m4
    m1 = xv < first_mean
    m2 = (~m1) & (i_l >= n_real)
    m3 = (~m1) & (~m2) & (cent_mean[i_l_c] == xv)
    res_multi = torch.where(m1, res_m1,
                torch.where(m2, res_m2,
                torch.where(m3, res_m3, res_m4)))
    # multi-centroid extremes
    res_multi = torch.where(xv >= t_max, torch.ones_like(res_multi),
                torch.where(xv <= t_min, torch.zeros_like(res_multi), res_multi))

    # single-centroid: linear ramp between t_min and t_max (0.5 for a degenerate range)
    rng = t_max - t_min
    res_single = torch.where(xv < t_min, torch.zeros_like(xv),
                 torch.where(xv > t_max, torch.ones_like(xv),
                 torch.where(rng < eps, torch.full_like(xv, 0.5),
                             (xv - t_min) / torch.where(rng == 0, 1.0, rng))))

    out = torch.where(n_real <= 1, res_single, res_multi)
    # empty digest -> nan, matching the original behavior
    out = torch.where(n_real == 0, torch.full_like(out, float("nan")), out)
    return out


def tdigest_quantile_tensor(qv, cent_mean, cent_weight, t_min, t_max, n_real):
    """Evaluate the TDigest inverse CDF (quantile function) at every element of ``qv`` for a single variable.

    Branchless, fixed-shape reformulation of the original masked implementation for ``torch.vmap``
    compatibility. See ``tdigest_cdf_tensor`` for the padding/``n_real`` convention.

    Args:
        qv (torch.Tensor): quantiles in ``[0, 1]``, any shape.
        cent_mean (torch.Tensor): padded ``(K,)`` centroid means, ascending, padding = ``+inf``.
        cent_weight (torch.Tensor): padded ``(K,)`` centroid weights, padding = ``0``.
        t_min, t_max: scalar min/max of the digest.
        n_real: number of valid centroids.

    Returns:
        torch.Tensor: interpolated values with the same shape as ``qv``.
    """
    K = cent_mean.numel()
    cum_sum = torch.cumsum(cent_weight, dim=0)
    cent_merged_weight = cum_sum - (cent_weight / 2.0)
    total_weight = cent_weight.sum()
    nlast = torch.clamp(n_real - 1, min=0)

    target_weight = qv * total_weight
    # padding merged-weights equal total_weight, so idx_r lands at n_real once past the last real centroid
    idx_r = _bucketize(cent_merged_weight, target_weight, "left")  # in [0, K]
    idx_l = idx_r - 1
    at_end = idx_r >= n_real
    idx_r_c = torch.minimum(idx_r, nlast)
    idx_l_c = torch.clamp(idx_l, min=0)

    # left boundary: (0, t_min) when idx_r == 0, else the merged-weight/mean of the left centroid
    x0 = torch.where(idx_r == 0, torch.zeros_like(target_weight), cent_merged_weight[idx_l_c])
    y0 = torch.where(idx_r == 0, t_min + torch.zeros_like(target_weight), cent_mean[idx_l_c])
    # right boundary: (total_weight, t_max) past the last centroid, else the merged-weight/mean of the right centroid
    x1 = torch.where(at_end, total_weight + torch.zeros_like(target_weight), cent_merged_weight[idx_r_c])
    y1 = torch.where(at_end, t_max + torch.zeros_like(target_weight), cent_mean[idx_r_c])

    denom = x1 - x0
    res = y0 + (target_weight - x0) * (y1 - y0) / torch.where(denom == 0, 1e-9, denom)

    # single centroid: interior quantiles all map to the single mean
    res = torch.where(n_real <= 1, cent_mean[0] + torch.zeros_like(qv), res)

    out = torch.where(qv <= 0, t_min + torch.zeros_like(qv),
          torch.where(qv >= 1, t_max + torch.zeros_like(qv), res))
    # empty digest -> nan, matching the original behavior
    out = torch.where(n_real == 0, torch.full_like(out, float("nan")), out)
    return out


def compute_pchip_slopes(x, y):
    """Vectorized Fritsch-Carlson monotone-cubic slopes (matching scipy's PchipInterpolator).

    Computes the derivative at every knot for a shape-preserving monotone cubic Hermite spline, for
    each variable (row) of ``x``/``y`` at once. Interior slopes use the weighted-harmonic-mean rule
    (set to 0 at local extrema so the spline never overshoots); endpoints use scipy's non-centered
    three-point edge formula with the same monotonicity limiting.

    Args:
        x (numpy.ndarray): strictly-increasing knot locations, shape ``(n_vars, M)``.
        y (numpy.ndarray): knot values, shape ``(n_vars, M)``.

    Returns:
        numpy.ndarray: knot slopes, shape ``(n_vars, M)``.
    """
    n, M = x.shape
    m = np.zeros_like(x)
    # tied knots (bumped by a tiny epsilon at saturated tails) produce huge/degenerate secants; the results
    # there are discarded by the monotonicity masks below, so suppress the expected divide/invalid warnings.
    with np.errstate(divide="ignore", invalid="ignore"):
        h = np.diff(x, axis=1)
        delta = np.diff(y, axis=1) / h
        if M < 3:
            m[:, 0] = delta[:, 0]
            m[:, -1] = delta[:, -1]
            return m
        hk = h[:, 1:]
        hk1 = h[:, :-1]
        dk = delta[:, 1:]
        dk1 = delta[:, :-1]
        w1 = 2 * hk + hk1
        w2 = hk + 2 * hk1
        m_int = (w1 + w2) / (w1 / dk1 + w2 / dk)
        mono = (dk1 * dk) > 0  # only assign a nonzero slope where the secants agree in sign
        m[:, 1:-1] = np.where(mono, m_int, 0.0)

        def edge(h0, h1, d0, d1):
            me = ((2 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
            me = np.where(np.sign(me) != np.sign(d0), 0.0, me)
            limit = (np.sign(d0) != np.sign(d1)) & (np.abs(me) > 3 * np.abs(d0))
            return np.where(limit, 3 * d0, me)

        m[:, 0] = edge(h[:, 0], h[:, 1], delta[:, 0], delta[:, 1])
        m[:, -1] = edge(h[:, -1], h[:, -2], delta[:, -1], delta[:, -2])
    return m


def pchip_eval_tensor(x_knots, z_knots, m_knots, xv):
    """Per-variable monotone cubic-Hermite (PCHIP) evaluation of the fitted transform (fast path).

    Written branchlessly / fixed-shape so it composes with ``torch.vmap`` over the channel dimension.
    Locates each element of ``xv`` among the ascending ``x_knots`` (via ``_bucketize`` for MPS
    friendliness), then evaluates the Hermite cubic using the precomputed knot slopes ``m_knots``.
    Values outside ``[x_knots[0], x_knots[-1]]`` saturate to the end z-knots, which equal
    ``dist_ppf(min_val)`` / ``dist_ppf(max_val)`` -- matching the exact path's ``min_val``/``max_val``
    clamping.

    Args:
        x_knots (torch.Tensor): ascending knot locations for one variable, shape ``(M,)``.
        z_knots (torch.Tensor): transformed value at each knot, shape ``(M,)`` (shared across vars).
        m_knots (torch.Tensor): PCHIP slope at each knot for one variable, shape ``(M,)``.
        xv (torch.Tensor): values to transform, any shape.

    Returns:
        torch.Tensor: transformed values with the same shape as ``xv``.
    """
    M = x_knots.shape[0]
    i = torch.clamp(_bucketize(x_knots, xv, "left"), 1, M - 1)
    x0 = x_knots[i - 1]
    x1 = x_knots[i]
    y0 = z_knots[i - 1]
    y1 = z_knots[i]
    m0 = m_knots[i - 1]
    m1 = m_knots[i]
    h = x1 - x0
    t = (xv - x0) / torch.where(h == 0, torch.ones_like(h), h)
    t2 = t * t
    t3 = t2 * t
    h00 = 2 * t3 - 3 * t2 + 1
    h10 = t3 - 2 * t2 + t
    h01 = -2 * t3 + 3 * t2
    h11 = t3 - t2
    z = h00 * y0 + h10 * h * m0 + h01 * y1 + h11 * h * m1
    z = torch.where(xv <= x_knots[0], z_knots[0], z)
    z = torch.where(xv >= x_knots[-1], z_knots[-1], z)
    return z


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
        compile: If True, transform/inverse_transform run through a cached ``torch.compile(fullgraph=True)`` of the
            vmapped per-variable kernel, which fuses the many elementwise ops for a sizable speedup on CPU/CUDA.
            Compilation is skipped on MPS (its inductor backend is immature and the ``_bucketize`` fallback already
            handles MPS); the default False preserves the plain eager-vmap behavior.
        fast_transform: If True, ``transform`` and ``inverse_transform`` use a monotone cubic (PCHIP) approximation
            of the fitted mapping instead of the exact TDigest CDF/quantile evaluation. Knots are placed at
            ``n_knots`` levels spaced uniformly in the output space; each direction is interpolated with a single
            searchsorted over ``n_knots`` (far fewer ops than the full kernel) for roughly a 3x speedup, at the cost
            of a small approximation error that is typically well below the TDigest's own error (see
            scripts/quantile_lut_experiment.py). Default False.
        n_knots: Number of PCHIP knots per variable when ``fast_transform`` is enabled (default 256). Ignored
            otherwise. Must be >= 4.
    """
    def __init__(self, compression=250, distribution="uniform", min_val=0.0000001, max_val=0.9999999,
                 channels_last=True, compile=False, fast_transform=False, n_knots=256):
        self.compression = compression
        self.distribution = distribution
        self.min_val = min_val
        self.max_val = max_val
        self.compile = compile
        self.fast_transform = fast_transform
        self.n_knots = n_knots
        self.centroids_ = None
        self.size_ = None
        self.min_ = None
        self.max_ = None
        self.centroids_mean_tensor = None
        self.centroids_weight_tensor = None
        self.min_tensor = None
        self.max_tensor = None
        # Padded/stacked centroid tensors used by the vmap-parallelized transforms.
        self.centroids_mean_stacked = None    # (n_vars, max_centroids), padding means = +inf
        self.centroids_weight_stacked = None  # (n_vars, max_centroids), padding weights = 0
        self.centroids_count = None           # (n_vars,) number of real centroids per variable
        # PCHIP knots for the optional fast_transform path; derived from the digest, rebuilt lazily,
        # invalidated on every fit/merge (see tensorize_attributes / ensure_pchip_knots).
        self.knots_x_ = None                  # (n_vars, n_knots) ascending knot locations
        self.knots_z_ = None                  # (n_knots,) transformed value at each knot (shared)
        self.knots_m_ = None                  # (n_vars, n_knots) forward PCHIP slopes dz/dx at the knots
        self.knots_minv_ = None               # (n_vars, n_knots) inverse PCHIP slopes dx/dz at the knots

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
        self.build_stacked_centroids()
        # invalidate any fast_transform knots; they are rebuilt lazily from the updated digest
        self.knots_x_ = None
        self.knots_z_ = None
        self.knots_m_ = None
        self.knots_minv_ = None
        return

    def build_stacked_centroids(self):
        """Pad the ragged per-variable centroid lists into rectangular tensors for ``torch.vmap``.

        Each variable's TDigest has a different number of centroids, so the per-variable ``(k_i,)`` mean/weight
        tensors cannot be batched directly. This pads every variable to ``K = max(k_i)`` columns, filling padding
        means with ``+inf`` (so they sort past all real centroids and are never selected by ``searchsorted``) and
        padding weights with ``0`` (so they contribute nothing to the cumulative/total weights). ``centroids_count``
        records the real centroid count per variable so the transform kernels know where the padding begins.
        """
        means = self.centroids_mean_tensor
        weights = self.centroids_weight_tensor
        counts = [m.shape[0] for m in means]
        n_vars = len(means)
        max_k = max(counts) if counts else 0
        dtype = means[0].dtype if n_vars else torch.float64
        mean_stacked = torch.full((n_vars, max_k), float("inf"), dtype=dtype)
        weight_stacked = torch.zeros((n_vars, max_k), dtype=dtype)
        for i, (m, w) in enumerate(zip(means, weights)):
            k = m.shape[0]
            mean_stacked[i, :k] = m
            weight_stacked[i, :k] = w
        self.centroids_mean_stacked = mean_stacked
        self.centroids_weight_stacked = weight_stacked
        self.centroids_count = torch.tensor(counts, dtype=torch.long)
        return

    def ensure_stacked_centroids(self):
        """Rebuild the stacked centroid tensors if missing (e.g. loaded from an older serialized scaler)."""
        if self.centroids_mean_stacked is None and self.centroids_mean_tensor is not None:
            self.build_stacked_centroids()
        return

    def build_pchip_knots(self):
        """Build per-variable monotone-cubic (PCHIP) knots approximating the exact transform (fast path).

        Knots are spaced uniformly in the *output* ``z`` over ``[dist_ppf(min_val), dist_ppf(max_val)]`` rather than
        in probability, which bounds the per-bin output error (each bin spans a fixed z-height) regardless of the
        input distribution -- equal-probability spacing badly under-resolves the steep tails. Each z-knot is mapped
        to a probability via the distribution CDF, and the x-knot is the digest quantile at that probability (via
        ``tdigest_quantile_tensor``). Together with the PCHIP slopes these let ``transform`` interpolate ``x -> z``
        with a single searchsorted over ``n_knots``, instead of the full TDigest CDF evaluation. Cheap and one-time;
        rebuilt lazily and invalidated on every fit/merge.
        """
        assert self.is_fit(), "Scaler has not been fit."
        assert self.n_knots >= 4, "fast_transform requires n_knots >= 4."
        self.ensure_stacked_centroids()
        M = int(self.n_knots)
        dtype = self.centroids_mean_stacked.dtype
        # z-knots: uniform in output space between the distribution ppf of min_val and max_val; p = dist_cdf(z)
        if self.distribution == "normal":
            z_lo, z_hi = torch.special.ndtri(torch.tensor([self.min_val, self.max_val], dtype=dtype))
            z_knots = torch.linspace(float(z_lo), float(z_hi), M, dtype=dtype)
            p = torch.special.ndtr(z_knots)
        elif self.distribution == "logistic":
            z_lo, z_hi = torch.logit(torch.tensor([self.min_val, self.max_val], dtype=dtype))
            z_knots = torch.linspace(float(z_lo), float(z_hi), M, dtype=dtype)
            p = torch.sigmoid(z_knots)
        else:  # uniform: transform output is the CDF itself, so z == p
            z_knots = torch.linspace(self.min_val, self.max_val, M, dtype=dtype)
            p = z_knots.clone()
        p = torch.clamp(p, self.min_val, self.max_val)
        # x-knots: digest quantile of each probability level, computed for every variable at once via vmap
        qfn = torch.vmap(tdigest_quantile_tensor, in_dims=(None, 0, 0, 0, 0, 0))
        x_knots = qfn(p,
                      self.centroids_mean_stacked,
                      self.centroids_weight_stacked,
                      self.min_tensor.to(dtype),
                      self.max_tensor.to(dtype),
                      self.centroids_count)
        x_np = x_knots.cpu().numpy()
        # enforce strictly-increasing x per variable (quantiles can tie at saturated tails), which PCHIP requires
        x_np = np.maximum.accumulate(x_np, axis=1)
        ties = np.diff(x_np, axis=1) <= 0
        if ties.any():
            x_np[:, 1:][ties] = x_np[:, :-1][ties] + 1e-12
        z_np = np.broadcast_to(z_knots.cpu().numpy(), x_np.shape).copy()
        # forward slopes dz/dx (transform) and inverse slopes dx/dz (inverse_transform); z-knots are strictly
        # increasing so the inverse fit needs no de-duplication, while x-knots were made monotone above.
        m_np = compute_pchip_slopes(x_np, z_np)
        minv_np = compute_pchip_slopes(z_np, x_np)
        self.knots_x_ = torch.from_numpy(np.ascontiguousarray(x_np))
        self.knots_z_ = z_knots.to(torch.float64)
        self.knots_m_ = torch.from_numpy(np.ascontiguousarray(m_np))
        self.knots_minv_ = torch.from_numpy(np.ascontiguousarray(minv_np))
        return

    def ensure_pchip_knots(self):
        """Build the fast_transform knots if missing (lazily, and after load / refit / merge)."""
        if self.knots_x_ is None and self.is_fit():
            self.build_pchip_knots()
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

    def _gather_scales(self, x_col_order, xv):
        """Select and stack the padded centroid parameters for the requested variables onto ``xv``'s device.

        Returns per-variable tensors aligned with the channel order of ``xv`` (indexed by ``x_col_order``):
        stacked means ``(n_sel, K)``, stacked weights ``(n_sel, K)``, min ``(n_sel,)``, max ``(n_sel,)``, and
        real-centroid counts ``(n_sel,)`` — ready to be mapped over dim 0 by ``torch.vmap``.
        """
        self.ensure_stacked_centroids()
        order = torch.as_tensor(x_col_order, dtype=torch.long)
        mean = self.centroids_mean_stacked.index_select(0, order).to(xv.device, dtype=xv.dtype)
        weight = self.centroids_weight_stacked.index_select(0, order).to(xv.device, dtype=xv.dtype)
        t_min = self.min_tensor.index_select(0, order).to(xv.device, dtype=xv.dtype)
        t_max = self.max_tensor.index_select(0, order).to(xv.device, dtype=xv.dtype)
        n_real = self.centroids_count.index_select(0, order).to(xv.device)
        return mean, weight, t_min, t_max, n_real

    def _get_batched_fn(self, kind, channel_dim, device_type):
        """Build (and cache) the vmapped per-variable transform, optionally wrapped in ``torch.compile``.

        The per-variable kernel is mapped over the channel dimension (stacked params on dim 0, xv on
        ``channel_dim``). When ``self.compile`` is set and the data is not on MPS, the vmapped function is wrapped
        in ``torch.compile(fullgraph=True)`` so inductor fuses the many elementwise ops into a few kernels. The
        cache is keyed by ``(kind, channel_dim, use_compile)`` so channels-first/last and compiled/eager variants
        coexist, and lives in a module-level weak map so nothing non-serializable lands on the scaler.
        """
        use_compile = bool(self.compile) and device_type != "mps"
        key = (kind, channel_dim, use_compile)
        cache = _BATCHED_CACHE.get(self)
        if cache is None:
            cache = {}
            _BATCHED_CACHE[self] = cache
        batched = cache.get(key)
        if batched is None:
            if kind == "transform":
                base = partial(transform_variable_tensor, min_val=self.min_val, max_val=self.max_val,
                               distribution=self.distribution)
            else:
                base = partial(inv_transform_variable_tensor, distribution=self.distribution)
            batched = torch.vmap(base, in_dims=(0, 0, 0, 0, 0, channel_dim), out_dims=channel_dim)
            if use_compile:
                batched = torch.compile(batched, fullgraph=True)
            cache[key] = batched
        return batched

    def _get_fast_fn(self, kind, channel_dim, device_type):
        """Build (and cache) the vmapped PCHIP evaluator for the fast_transform path.

        Mirrors ``_get_batched_fn`` but maps ``pchip_eval_tensor`` over the channel dimension. The search/output
        roles swap between directions: for ``"transform"`` the search knots are the per-variable x-knots and the
        outputs are the shared z-knots; for ``"inverse"`` the search knots are the shared z-knots and the outputs
        are the per-variable x-knots. ``in_dims`` places the per-variable arg on dim 0 and the shared arg at
        ``None`` accordingly. Optionally torch.compiled off MPS.
        """
        use_compile = bool(self.compile) and device_type != "mps"
        # namespaced ("fast_*") so these keys never collide with _get_batched_fn's ("transform"/"inverse")
        # entries in the shared _BATCHED_CACHE
        key = ("fast_" + kind, channel_dim, use_compile)
        cache = _BATCHED_CACHE.get(self)
        if cache is None:
            cache = {}
            _BATCHED_CACHE[self] = cache
        batched = cache.get(key)
        if batched is None:
            # transform: (x_knots per-var, z_knots shared, slopes per-var, data on channel_dim)
            # inverse:   (z_knots shared, x_knots per-var, slopes per-var, data on channel_dim)
            in_dims = (0, None, 0, channel_dim) if kind == "transform" else (None, 0, 0, channel_dim)
            batched = torch.vmap(pchip_eval_tensor, in_dims=in_dims, out_dims=channel_dim)
            if use_compile:
                batched = torch.compile(batched, fullgraph=True)
            cache[key] = batched
        return batched

    def transform(self, x, channels_last=None):
        xv, x_transformed, channels_last, channel_dim, x_col_order = self.process_x_for_transform(x, channels_last)
        if self.fast_transform:
            # Approximate path: interpolate the fitted transform with a per-variable monotone cubic (PCHIP),
            # replacing the exact TDigest CDF evaluation with a single searchsorted over n_knots. Knots follow
            # the requested channel order (x_col_order) and are moved onto xv's device/dtype.
            self.ensure_pchip_knots()
            order = torch.as_tensor(x_col_order, dtype=torch.long)
            xk = self.knots_x_.index_select(0, order).to(xv.device, dtype=xv.dtype)
            mk = self.knots_m_.index_select(0, order).to(xv.device, dtype=xv.dtype)
            zk = self.knots_z_.to(xv.device, dtype=xv.dtype)
            batched = self._get_fast_fn("transform", channel_dim, xv.device.type)
            x_transformed = batched(xk, zk, mk, xv)
            return self.package_transformed_x(x_transformed, x)
        mean, weight, t_min, t_max, n_real = self._gather_scales(x_col_order, xv)
        # vmap parallelizes the per-variable transform over the channel dimension, which is far faster than a
        # Python loop; when compile is enabled the fused torch.compile variant is used (see _get_batched_fn).
        batched = self._get_batched_fn("transform", channel_dim, xv.device.type)
        x_transformed = batched(mean, weight, t_min, t_max, n_real, xv)

        x_transformed_final = self.package_transformed_x(x_transformed, x)
        return x_transformed_final

    def fit_transform(self, x, channels_last=None, weight=None):
        self.fit(x, weight=weight)
        return self.transform(x, channels_last=channels_last)

    def inverse_transform(self, x, channels_last=None):
        xv, x_transformed, channels_last, channel_dim, x_col_order = self.process_x_for_transform(x, channels_last)
        if self.fast_transform:
            # Approximate path: interpolate the fitted inverse with a per-variable monotone cubic (PCHIP),
            # searching the input z-values in the shared z-knots and interpolating to the per-variable x-knots
            # (roles swapped relative to transform). Uses the precomputed dx/dz slopes (knots_minv_).
            self.ensure_pchip_knots()
            order = torch.as_tensor(x_col_order, dtype=torch.long)
            zk = self.knots_z_.to(xv.device, dtype=xv.dtype)
            xk = self.knots_x_.index_select(0, order).to(xv.device, dtype=xv.dtype)
            mk = self.knots_minv_.index_select(0, order).to(xv.device, dtype=xv.dtype)
            batched = self._get_fast_fn("inverse", channel_dim, xv.device.type)
            x_transformed = batched(zk, xk, mk, xv)
            return self.package_transformed_x(x_transformed, x)
        mean, weight, t_min, t_max, n_real = self._gather_scales(x_col_order, xv)

        batched = self._get_batched_fn("inverse", channel_dim, xv.device.type)
        x_transformed = batched(mean, weight, t_min, t_max, n_real, xv)

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