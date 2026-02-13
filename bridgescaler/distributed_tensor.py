from . import require_torch
require_torch()   # enforce torch availability/version at import time
import torch

from copy import deepcopy


class DBaseScalerTensor:
    """
    Base distributed scaler class for tensor. Used only to store attributes and methods shared across all distributed
    scaler subclasses.
    """

    def __init__(self, channels_last=True):
        self.x_columns_ = None
        self._fit = False
        self.channels_last = channels_last

    def is_fit(self):
        return self._fit

    @staticmethod
    def extract_x_columns(x, channels_last=True):
        """
        Extract the variable (column) names from input x

        The variable/channel names are stored as a list of strings in the
        `variable_names` attribute. If this attribute does not exist, an error will be raised.
        This extraction assumes that the channels are located in the last dimension.

        Args:
            x (torch.tensor): tensor of values.
            channels_last (bool): If True, then assume the variable or channel dimension is the last dimension of the
                array. If False, then assume the variable or channel dimension is second.

        Returns:
            x_columns (torch.tensor): list of strings with variable names
        """
        var_dim_num = -1
        if not channels_last:
            var_dim_num = 1
        assert isinstance(x, torch.Tensor), "Input must be a PyTorch tensor"
        if hasattr(x, 'variable_names'):
            x_columns = x.variable_names
        else:
            x_columns = list(range(x.shape[var_dim_num]))
        #assert getattr(x, 'variable_names', None) is not None, "variable_names attribute is missing or empty"
        assert len(x_columns) == len(
            set(x_columns)), f"Duplicates found! Unique count: {len(set(x_columns))}, Total count: {len(x_columns)}"
        return x_columns

    @staticmethod
    def extract_array(x):
        pass

    def get_column_order(self, x_in_columns):
        """
        Get the indices of the scaler columns that have the same name as the variables (columns) in the input x tensor. This
        enables users to pass a torch.Tensor to transform or inverse_transform with fewer variables than
        the original scaler or variables in a different order and still have the input dataset be transformed properly.

        Args:
            x_in_columns (list): list of input variable names

        Returns:
            x_in_col_indices (torch.Tensor): indices of the input variables from x in the scaler in order.
        """
        assert all(var in self.x_columns_ for var in x_in_columns), "Some input variables not in scaler x_columns."
        x_in_col_indices = [self.x_columns_.index(item) for item in x_in_columns if item in self.x_columns_]
        return x_in_col_indices

    @staticmethod
    def package_transformed_x(x_transformed, x):
        """
        Repackaged a transformed torch.Tensor into the same datatype as the original x, including
        all metadata.

        Args:
            x_transformed (torch.Tensor): array after being transformed or inverse transformed
            x (torch.Tensor)

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
        x_in_cols = self.extract_x_columns(x, channels_last=channels_last)
        assert (
            x.shape[channel_dim] == len(self.x_columns_)
        ), "Number of input columns does not match scaler."
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
    supports torch.tensor and returns a transformed tensor.
    """

    def __init__(self, channels_last=True):
        self.mean_x_ = None
        self.n_ = 0
        self.var_x_ = None
        super().__init__(channels_last=channels_last)

    def fit(self, x, weight=None):
        x_columns = self.extract_x_columns(x, channels_last=self.channels_last)
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
            ), "New data has a different number of columns"
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
            x (torch.tensor): Input data.
            channels_last: Override the default channels_last parameter of the scaler.

        Returns:
            x_transformed (torch.tensor): Transformed data in the same shape and type as x.
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
        assert (
            type(other) is DStandardScalerTensor
        ), "Input is not DStandardScalerTensor"
        assert (other.x_columns_ == self.x_columns_
        ), "Scaler columns do not match."
        current = deepcopy(self)
        current.mean_x_ = (self.n_ * self.mean_x_ + other.n_ * other.mean_x_) / (
            self.n_ + other.n_
        )
        combined_var = (self.n_ * self.var_x_ + other.n_ * other.var_x_) / (
            self.n_ + other.n_
        )
        combined_var_corr = (
            self.n_ * other.n_ * (self.mean_x_ - other.mean_x_) ** 2
        ) / ((self.n_ + other.n_) ** 2)
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
        x_columns = self.extract_x_columns(x, channels_last=self.channels_last)
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
            ), "New data has a different number of columns"
            x_col_order = self.get_column_order(x_columns)
            if self.channels_last:
                self.max_x_ = torch.maximum(
                    self.max_x_, torch.amax(xv, dim=tuple(range(xv.ndim - 1)))
                )
                self.min_x_ = torch.minimum(
                    self.min_x_, torch.amin(xv, dim=tuple(range(xv.ndim - 1)))
                )
            else:
                self.max_x_ = torch.maximum(
                    self.max_x_, torch.amax(xv, dim=tuple(d for d in range(xv.ndim) if d != 1)))
                self.min_x_ = torch.minimum(
                    self.min_x_, torch.amin(xv, dim=tuple(d for d in range(xv.ndim) if d != 1)))
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
        assert type(other) is DMinMaxScalerTensor, "Input is not DMinMaxScaler"
        assert other.x_columns_ == self.x_columns_, "Scaler columns do not match."
        current = deepcopy(self)
        current.max_x_ = torch.maximum(self.max_x_, other.max_x_)
        current.min_x_ = torch.minimum(self.min_x_, other.min_x_)
        return current
