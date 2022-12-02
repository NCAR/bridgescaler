from bridgescaler import save_scaler, load_scaler
from bridgescaler.backend import create_synthetic_data
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, QuantileTransformer
scaler_objs = {"StandardScaler": StandardScaler,
               "MinMaxScaler": MinMaxScaler,
               "RobustScaler": RobustScaler,
               "QuantileTransformer": QuantileTransformer}
from os.path import exists




def test_scaler_io():
    try:
        x_data = create_synthetic_data()
        for scaler_name, scaler_obj in scaler_objs.items():
            scaler = scaler_obj()
            x_scaled_data = scaler.fit_transform(x_data)
            save_scaler(scaler, "test.json")
            assert exists("test.json")
            loaded_scaler = load_scaler("test.json")
            assert type(loaded_scaler) == type(scaler)
            loaded_scaled_data = loaded_scaler.transform(x_data)
            assert np.max(np.abs(x_scaled_data - loaded_scaled_data)) < np.finfo(np.float32).eps, scaler_name + " transform does not match"
    finally:
        if exists("test.json"):
            os.remove("test.json")
    return