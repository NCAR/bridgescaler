from bridgescaler import save_scaler, load_scaler, print_scaler, read_scaler
from bridgescaler.backend import create_synthetic_data
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, QuantileTransformer
from bridgescaler.distributed import DStandardScaler, DMinMaxScaler, DQuantileScaler
from pandas import DataFrame
from os.path import exists


scaler_objs = {"StandardScaler": StandardScaler,
               "MinMaxScaler": MinMaxScaler,
               "RobustScaler": RobustScaler,
               "QuantileTransformer": QuantileTransformer,
               "DStandardScaler": DStandardScaler,
               "DMinMaxScaler": DMinMaxScaler,
               "DQuantileTransformer": DQuantileTransformer,
               "DQuantileScaler": DQuantileScaler}


def test_scaler_io():
    try:
        x_data = create_synthetic_data()
        for scaler_name, scaler_obj in scaler_objs.items():
            scaler = scaler_obj()
            x_scaled_data = scaler.fit_transform(x_data)
            save_scaler(scaler, "test.json")
            assert exists("test.json")
            loaded_scaler = load_scaler("test.json")
            assert type(loaded_scaler) is type(scaler), "Type mismatch"
            loaded_scaled_data = loaded_scaler.transform(x_data)
            if type(x_scaled_data) is DataFrame:
                transform_diff = np.max(np.abs(x_scaled_data.values - loaded_scaled_data.values))
            else:
                transform_diff = np.max(np.abs(x_scaled_data - loaded_scaled_data))

            assert transform_diff < np.finfo(np.float32).eps, scaler_name + " transform does not match"
    finally:
        if exists("test.json"):
            os.remove("test.json")
    return


def test_scaler_str():
    x_data = create_synthetic_data()
    for scaler_name, scaler_obj in scaler_objs.items():
        scaler = scaler_obj()
        x_scaled_data = scaler.fit_transform(x_data)
        scaler_str = print_scaler(scaler)
        loaded_scaler = read_scaler(scaler_str)
        assert type(loaded_scaler) is type(scaler), "Type Mismatch"
        loaded_scaled_data = loaded_scaler.transform(x_data)
        if type(x_scaled_data) is DataFrame:
            transform_diff = np.max(np.abs(x_scaled_data.values - loaded_scaled_data.values))
        else:
            transform_diff = np.max(np.abs(x_scaled_data - loaded_scaled_data))
        assert transform_diff < np.finfo(np.float32).eps, scaler_name + " transform does not match"
