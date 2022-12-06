from bridgescaler.group import GroupStandardScaler
from bridgescaler.backend import create_synthetic_data
from bridgescaler import save_scaler, load_scaler
import numpy as np

def test_group_standard_scaler():
    x_data = create_synthetic_data()
    groups = [["A", "B"], "C", "D"]
    save_filename = "group_test.json"
    group_scaler = GroupStandardScaler()
    transformed_x = group_scaler.fit_transform(x_data, groups)
    assert transformed_x.shape == x_data.shape
    inverse_x = group_scaler.inverse_transform(transformed_x)
    assert inverse_x.shape == x_data.shape
    save_scaler(group_scaler, save_filename)
    reloaded_scaler = load_scaler(save_filename)
    reloaded_scale_x = reloaded_scaler.transform(x_data)
    assert np.all(transformed_x == reloaded_scale_x)
    return