from copy import deepcopy
import importlib.util

from packaging import version
import torch

REQUIRED_VERSION = "2.8.0"  # required torch version

# Check if PyTorch is installed
if importlib.util.find_spec("torch") is None:
    raise ImportError("PyTorch is not installed")

installed_version = torch.__version__

# Validate version
if version.parse(installed_version) != version.parse(REQUIRED_VERSION):
    raise RuntimeError(
        f"PyTorch version mismatch: required {REQUIRED_VERSION}, "
        f"found {installed_version}"
    )


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
        Extract column indices to be transformed from x. All of these assume that the columns are in the last dimension.

        Args:
            x (torch.tensor): tensor of values to be transformed.
            channels_last (bool): If True, then assume the variable or channel dimension is the last dimension of the
                array. If False, then assume the variable or channel dimension is second.

        Returns:
            x_columns (torch.tensor): tensor of column indices.
        """
        var_dim_num = -1
        if not channels_last:
            var_dim_num = 1
        assert isinstance(x, torch.Tensor), "Input must be a PyTorch tensor"
        x_columns = torch.arange(x.shape[var_dim_num])
        return x_columns

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
        assert (
            x.shape[channel_dim] == self.x_columns_.shape[0]
        ), "Number of input columns does not match scaler."
        x_col_order = torch.arange(x.shape[channel_dim])
        xv = x
        x_transformed = torch.zeros(xv.shape, dtype=xv.dtype)
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
                    self.n_ += torch.prod(torch.tensor(xv.shape[:-1]))
                else:
                    self.n_ += xv.shape[0] * \
                        torch.prod(torch.tensor(xv.shape[2:]))
            else:
                self.n_ += xv.shape[0]
            self.mean_x_ = torch.zeros(xv.shape[channel_dim], dtype=xv.dtype)
            self.var_x_ = torch.zeros(xv.shape[channel_dim], dtype=xv.dtype)

            if self.channels_last:
                for i in range(xv.shape[channel_dim]):
                    self.mean_x_[i] = torch.mean(xv[..., i])
                    self.var_x_[i] = torch.var(xv[..., i], unbiased=False)
            else:
                for i in range(xv.shape[channel_dim]):
                    self.mean_x_[i] = torch.mean(xv[:, i])
                    self.var_x_[i] = torch.var(xv[:, i], unbiased=False)

        else:
            # Update existing scaler with new data
            assert (
                x.shape[channel_dim] == self.x_columns_.shape[0]
            ), "New data has a different number of columns"
            if self.channels_last:
                x_col_order = torch.arange(x.shape[-1])
            else:
                x_col_order = torch.arange(x.shape[1])
            if len(xv.shape) > 2:
                if self.channels_last:
                    new_n = torch.prod(torch.tensor(xv.shape[:-1]))
                else:
                    new_n = xv.shape[0] * \
                        torch.prod(torch.tensor(xv.shape[2:]))
            else:
                new_n = xv.shape[0]
            for i, o in enumerate(x_col_order):
                if self.channels_last:
                    new_mean = torch.mean(xv[..., i])
                    new_var = torch.var(xv[..., i], unbiased=False)
                else:
                    new_mean = torch.mean(xv[:, i])
                    new_var = torch.var(xv[:, i], unbiased=False)
                combined_mean = (self.n_ * self.mean_x_[o] + new_n * new_mean) / (
                    self.n_ + new_n
                )
                weighted_var = (self.n_ * self.var_x_[o] + new_n * new_var) / (
                    self.n_ + new_n
                )
                var_correction = (
                    self.n_ * new_n * (self.mean_x_[o] - new_mean) ** 2
                ) / ((self.n_ + new_n) ** 2)
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
        x_mean, x_var = self.get_scales()
        if channels_last:
            for i, o in enumerate(x_col_order):
                x_transformed[..., i] = (
                    xv[..., i] - x_mean[o]) / torch.sqrt(x_var[o])
        else:
            for i, o in enumerate(x_col_order):
                x_transformed[:, i] = (
                    xv[:, i] - x_mean[o]) / torch.sqrt(x_var[o])
        return x_transformed

    def inverse_transform(self, x, channels_last=None):
        (
            xv,
            x_transformed,
            channels_last,
            channel_dim,
            x_col_order,
        ) = self.process_x_for_transform(x, channels_last)
        x_mean, x_var = self.get_scales()
        if channels_last:
            for i, o in enumerate(x_col_order):
                x_transformed[..., i] = xv[..., i] * \
                    torch.sqrt(x_var[o]) + x_mean[o]
        else:
            for i, o in enumerate(x_col_order):
                x_transformed[:, i] = xv[:, i] * \
                    torch.sqrt(x_var[o]) + x_mean[o]
        return x_transformed

    def get_scales(self):
        return self.mean_x_, self.var_x_

    def __add__(self, other):
        assert (
            type(other) is DStandardScalerTensor
        ), "Input is not DStandardScalerTensor"
        assert torch.all(
            other.x_columns_ == self.x_columns_
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
    the mins and maxes as a reduction step. Scaler
    supports torch.tensor and will return a transformed array in the
    same form as the original with column or coordinate names preserved.
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
            self.max_x_ = torch.zeros(xv.shape[channel_dim], dtype=xv.dtype)
            self.min_x_ = torch.zeros(xv.shape[channel_dim], dtype=xv.dtype)

            if self.channels_last:
                for i in range(xv.shape[channel_dim]):
                    self.max_x_[i] = torch.max(xv[..., i])
                    self.min_x_[i] = torch.min(xv[..., i])
            else:
                for i in range(xv.shape[channel_dim]):
                    self.max_x_[i] = torch.max(xv[:, i])
                    self.min_x_[i] = torch.min(xv[:, i])
        else:
            # Update existing scaler with new data
            assert (
                x.shape[channel_dim] == self.x_columns_.shape[0]
            ), "New data has a different number of columns"
            if self.channels_last:
                x_col_order = torch.arange(x.shape[-1])
            else:
                x_col_order = torch.arange(x.shape[1])
            if self.channels_last:
                for i, o in enumerate(x_col_order):
                    self.max_x_[o] = torch.maximum(
                        self.max_x_[o], torch.max(xv[..., i])
                    )
                    self.min_x_[o] = torch.minimum(
                        self.min_x_[o], torch.min(xv[..., i])
                    )
            else:
                for i, o in enumerate(xv.shape[channel_dim]):
                    self.max_x_[o] = torch.maximum(
                        self.max_x_[o], torch.max(xv[:, i]))
                    self.min_x_[o] = torch.minimum(
                        self.min_x_[o], torch.min(xv[:, i]))
        self._fit = True

    def transform(self, x, channels_last=None):
        (
            xv,
            x_transformed,
            channels_last,
            channel_dim,
            x_col_order,
        ) = self.process_x_for_transform(x, channels_last)
        if channels_last:
            for i, o in enumerate(x_col_order):
                x_transformed[..., i] = (xv[..., i] - self.min_x_[o]) / (
                    self.max_x_[o] - self.min_x_[o]
                )
        else:
            for i, o in enumerate(x_col_order):
                x_transformed[:, i] = (xv[:, i] - self.min_x_[o]) / (
                    self.max_x_[o] - self.min_x_[o]
                )
        return x_transformed

    def inverse_transform(self, x, channels_last=None):
        (
            xv,
            x_transformed,
            channels_last,
            channel_dim,
            x_col_order,
        ) = self.process_x_for_transform(x, channels_last)
        if channels_last:
            for i, o in enumerate(x_col_order):
                x_transformed[..., i] = (
                    xv[..., i] * (self.max_x_[o] - self.min_x_[o]
                                  ) + self.min_x_[o]
                )
        else:
            for i, o in enumerate(x_col_order):
                x_transformed[:, i] = (
                    xv[:, i] * (self.max_x_[o] - self.min_x_[o]) +
                    self.min_x_[o]
                )
        return x_transformed

    def get_scales(self):
        return self.min_x_, self.max_x_

    def __add__(self, other):
        assert type(other) is DMinMaxScalerTensor, "Input is not DMinMaxScaler"
        assert torch.all(
            other.x_columns_ == self.x_columns_
        ), "Scaler columns do not match."
        current = deepcopy(self)
        current.max_x_ = torch.maximum(self.max_x_, other.max_x_)
        current.min_x_ = torch.minimum(self.min_x_, other.min_x_)
        return current
