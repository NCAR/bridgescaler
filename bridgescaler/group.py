import numpy as np
from copy import deepcopy


class GroupBaseScaler:
    def __init__(self):
        self.groups_ = None
        self.group_index_ = None
        self.x_columns_ = None

    def fit(self, x, y=None, groups=None):
        self._fit(x, groups)
        return self

    def fit_transform(self, x, y=None, groups=None):
        self._fit(x, groups)
        return self.transform(x)

    def transform(self, x):
        transformed_x = deepcopy(x)
        is_df = hasattr(x, "columns")
        for column in self.x_columns_:
            group_index = self.find_group(column)
            if is_df:
                transformed_x.loc[:, column] = self._transform_column(x[column], group_index)
            else:
                transformed_x[:, column] = self._transform_column(x[:, column], group_index)
        return transformed_x

    def inverse_transform(self, x):
        transformed_x = deepcopy(x)
        is_df = hasattr(x, "columns")
        for column in self.x_columns_:
            group_index = self.find_group(column)
            if is_df:
                transformed_x.loc[:, column] = self._inverse_transform_column(x[column], group_index)
            else:
                transformed_x[:, column] = self._inverse_transform_column(x[:, column], group_index)
        return transformed_x

    def set_groups(self, x, groups):
        if groups is None:
            if hasattr(x, "columns"):
                self.groups_ = list(x.columns)
                self.x_columns_ = x.columns
            else:
                self.groups_ = list(range(x.shape[1]))
                self.x_columns_ = list(range(x.shape[1]))
        else:
            self.groups_ = groups
            if hasattr(x, "columns"):
                self.x_columns_ = x.columns
            else:
                self.x_columns_ = list(range(x.shape[1]))
        self.group_index_ = np.arange(len(self.groups_))

    def find_group(self, var_name):
        group_index = -1
        for g, group in enumerate(self.groups_):
            if type(group) != list and var_name == group:
                group_index = g
            elif var_name in group:
                group_index = g
        assert group_index >= 0, var_name + " not found in groups."
        return group_index

    def _fit(self, x, groups):
        raise NotImplementedError

    def _transform_column(self, x, group_index):
        raise NotImplementedError

    def _inverse_transform_column(self, x, group_index):
        raise NotImplementedError


class GroupStandardScaler(GroupBaseScaler):
    """
    Scaler that enables calculation and sharing of scaling parameters among multiple variables via variable groupings.
    This is useful for situations where variables are related, such as temperatures at different height levels.

    Groups are specified as a list of column ids, which can be column names for pandas dataframes or column indices
    for numpy arrays.

    For example:
    ```
    groups = [["a", "b"], ["c", "d"], "e"]
    ```
    "a" and "b" are a single group and all values of both will be included when calculating the mean and standard
     deviation for that group.

    """
    def __init__(self):
        self.center_ = None
        self.scale_ = None
        super().__init__()

    def _fit(self, x, groups):
        self.set_groups(x, groups)
        self.center_ = np.zeros(self.group_index_.shape)
        self.scale_ = np.zeros(self.group_index_.shape)
        is_df = hasattr(x, "columns")
        for g in self.group_index_:
            if is_df:
                self.center_[g] = np.mean(x[self.groups_[g]].values)
                self.scale_[g] = np.std(x[self.groups_[g]].values)
            else:
                self.center_[g] = np.mean(x[:, self.groups_[g]])
                self.scale_[g] = np.std(x[:, self.groups_[g]])

        return

    def _transform_column(self, x_column, group_index):
        return (x_column - self.center_[group_index]) / self.scale_[group_index]

    def _inverse_transform_column(self, x_column, group_index):
        return x_column * self.scale_[group_index] + self.center_[group_index]


class GroupMinMaxScaler(GroupBaseScaler):
    """
    Group version of MinMaxScaler
    """
    def __init__(self):
        self.mins_ = None
        self.maxes_ = None
        GroupBaseScaler.__init__(self)
        return

    def _fit(self, x, groups):
        self.set_groups(x, groups)
        self.mins_ = np.zeros(self.group_index_.shape)
        self.maxes_ = np.zeros(self.group_index_.shape)
        for g in self.group_index_:
            self.mins_[g] = np.min(x[self.groups_[g]])
            self.maxes_[g] = np.max(x[self.groups_[g]])
        return

    def _transform_column(self, x_column, group_index):
        return (x_column - self.mins_[group_index]) / (self.maxes_[group_index] - self.mins_[group_index])

    def _inverse_transform_column(self, x_column, group_index):
        return x_column * (self.maxes_[group_index] - self.mins_[group_index]) + self.mins_[group_index]


class GroupRobustScaler(GroupBaseScaler):
    """
    Group version of RobustScaler

    """
    def __init__(self, quartile_range=(25.0, 75.0)):
        self.quartile_range = quartile_range
        self.center_ = None
        self.scale_ = None
        super().__init__()

    def _fit(self, x, groups):
        self.set_groups(x, groups)
        self.center_ = np.zeros(self.group_index_.shape)
        self.scale_ = np.zeros(self.group_index_.shape)
        is_df = hasattr(x, "columns")
        for g in self.group_index_:
            if is_df:
                self.center_[g] = np.median(x[self.groups_[g]])
                self.scale_[g] = np.abs(np.quantile(x[self.groups_[g]], self.quartile_range[1] / 100.0) -
                                        np.quantile(x[self.groups_[g]], self.quartile_range[0] / 100.0))
            else:
                self.center_[g] = np.median(x[:, self.groups_[g]])
                self.scale_[g] = np.abs(np.quantile(x[:, self.groups_[g]], self.quartile_range[1] / 100.0) -
                                        np.quantile(x[:, self.groups_[g]], self.quartile_range[0] / 100.0))

    def _transform_column(self, x_column, group_index):
        return (x_column - self.center_[group_index]) / self.scale_[group_index]

    def _inverse_transform_column(self, x_column, group_index):
        return x_column * self.scale_[group_index] + self.center_[group_index]





