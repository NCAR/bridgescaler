from bridgescaler.group import GroupStandardScaler, GroupMinMaxScaler, GroupRobustScaler
from bridgescaler.backend import create_synthetic_data
from bridgescaler import save_scaler, load_scaler
import numpy as np
from os.path import exists
import os


def test_group_standard_scaler():
    try:
        x_data = create_synthetic_data()
        x_data_numpy = x_data.values
        groups = [["A", "B"], "C", "D"]
        n_groups = [[0, 1], 2, 3]
        save_filename = "group_test.json"
        group_scaler_n = GroupStandardScaler()
        n_transformed = group_scaler_n.fit_transform(x_data_numpy, n_groups)
        n_inv_transformed = group_scaler_n.inverse_transform(n_transformed)
        assert np.max(np.abs(n_inv_transformed - x_data_numpy)) < np.finfo(np.float32).eps
        group_scaler = GroupStandardScaler()
        transformed_x = group_scaler.fit_transform(x_data, groups)
        assert transformed_x.shape == x_data.shape
        inverse_x = group_scaler.inverse_transform(transformed_x)
        assert inverse_x.shape == x_data.shape
        assert np.max(np.abs(inverse_x.values - x_data.values)) < np.finfo(np.float32).eps
        save_scaler(group_scaler, save_filename)
        reloaded_scaler = load_scaler(save_filename)
        reloaded_scale_x = reloaded_scaler.transform(x_data)
        assert np.all(transformed_x == reloaded_scale_x)
    finally:
        if exists("group_test.json"):
            os.remove("group_test.json")
    return

def test_group_minmax_scaler():
    try:
        x_data = create_synthetic_data()
        x_data_numpy = x_data.values
        groups = [["A", "B"], "C", "D"]
        n_groups = [[0, 1], 2, 3]
        save_filename = "group_test.json"
        group_scaler_n = GroupMinMaxScaler()
        n_transformed = group_scaler_n.fit_transform(x_data_numpy, n_groups)
        n_inv_transformed = group_scaler_n.inverse_transform(n_transformed)
        assert np.max(np.abs(n_inv_transformed - x_data_numpy)) < np.finfo(np.float32).eps
        group_scaler = GroupMinMaxScaler()
        transformed_x = group_scaler.fit_transform(x_data, groups)
        assert transformed_x.shape == x_data.shape
        inverse_x = group_scaler.inverse_transform(transformed_x)
        assert inverse_x.shape == x_data.shape
        assert np.max(np.abs(inverse_x.values - x_data.values)) < np.finfo(np.float32).eps
        save_scaler(group_scaler, save_filename)
        reloaded_scaler = load_scaler(save_filename)
        reloaded_scale_x = reloaded_scaler.transform(x_data)
        assert np.all(transformed_x == reloaded_scale_x)
    finally:
        if exists("group_test.json"):
            os.remove("group_test.json")
    return


def test_group_robust_scaler():
    try:
        x_data = create_synthetic_data()
        x_data_numpy = x_data.values
        groups = [["A", "B"], "C", "D"]
        n_groups = [[0, 1], 2, 3]
        save_filename = "group_test.json"
        group_scaler_n = GroupRobustScaler()
        n_transformed = group_scaler_n.fit_transform(x_data_numpy, n_groups)
        n_inv_transformed = group_scaler_n.inverse_transform(n_transformed)
        assert np.max(np.abs(n_inv_transformed - x_data_numpy)) < np.finfo(np.float32).eps
        group_scaler = GroupRobustScaler()
        transformed_x = group_scaler.fit_transform(x_data, groups)
        assert transformed_x.shape == x_data.shape
        inverse_x = group_scaler.inverse_transform(transformed_x)
        assert inverse_x.shape == x_data.shape
        assert np.max(np.abs(inverse_x.values - x_data.values)) < np.finfo(np.float32).eps
        save_scaler(group_scaler, save_filename)
        reloaded_scaler = load_scaler(save_filename)
        reloaded_scale_x = reloaded_scaler.transform(x_data)
        assert np.all(transformed_x == reloaded_scale_x)
    finally:
        if exists("group_test.json"):
            os.remove("group_test.json")
    return