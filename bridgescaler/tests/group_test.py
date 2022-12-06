from bridgescaler.group import GroupStandardScaler
from bridgescaler.backend import create_synthetic_data
from bridgescaler import save_scaler, load_scaler
import numpy as np
from os.path import exists
import os


def test_group_standard_scaler():
    try:
        x_data = create_synthetic_data()
        groups = [["A", "B"], "C", "D"]
        save_filename = "group_test.json"
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