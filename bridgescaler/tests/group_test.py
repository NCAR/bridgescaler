from bridgescaler.group import GroupStandardScaler
from bridgescaler.backend import create_synthetic_data


def test_group_standard_scaler():
    x_data = create_synthetic_data()
    groups = [["A", "B"], "C", "D"]
    group_scaler = GroupStandardScaler()
    transformed_x = group_scaler.fit_transform(x_data, groups)
    assert transformed_x.shape == x_data.shape
    inverse_x = group_scaler.inverse_transform(transformed_x)
    assert inverse_x.shape == x_data.shape
    return