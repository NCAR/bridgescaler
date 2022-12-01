from bridgescaler import save_scaler, load_scaler
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, QuantileTransformer
scaler_objs = {"StandardScaler": StandardScaler,
               "MinMaxScaler": MinMaxScaler,
               "RobustScaler": RobustScaler,
               "QuantileTransformer": QuantileTransformer}
from os.path import exists


def test_scaler_io():
    locs = np.array([0, 5, -2, 350.5], dtype=np.float32)
    scales = np.array([1.0, 10, 0.1, 5000.0])
    names = ["A", "B", "C", "D"]
    num_examples = 205
    x_data_dict = {}
    for l in range(locs.shape[0]):
        x_data_dict[names[l]] = np.random.normal(loc=locs[l], scale=scales[l], size=num_examples)
    x_data = pd.DataFrame(x_data_dict)
    for scaler_name, scaler_obj in scaler_objs.items():
        scaler = scaler_obj()
        x_scaled_data = scaler.fit_transform(x_data)
        save_scaler(scaler, "test.json")
        assert exists("test.json")
        loaded_scaler = load_scaler("test.json")
        assert type(loaded_scaler) == type(scaler_obj)
        loaded_scaled_data = loaded_scaler.transform(x_data)
        assert np.max(np.abs(x_scaled_data - loaded_scaled_data)) < np.finfo(np.float32).eps, scaler_name + " transform does not match"
    return