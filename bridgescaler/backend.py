from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer
import numpy as np
import json

scaler_objs = {"StandardScaler": StandardScaler,
               "MinMaxScaler": MinMaxScaler,
               "RobustScaler": RobustScaler,
               "QuantileTransformer": QuantileTransformer}


def save_scaler(scaler, scaler_file):
    scaler_params = scaler.__dict__
    for k, v in scaler_params.items():
        if type(v) == np.ndarray:
            if v.dtype == "object":
                scaler_params[k] = v.tolist()
            else:
                scaler_params[k] = [x.item() for x in v]
    scaler_params["type"] = str(type(scaler))
    with open(scaler_file, "w") as file_obj:
        json.dump(scaler_params, file_obj, indent=4, sort_keys=True)
    return


def load_scaler(scaler_file):
    """
    Initialize scikit-learn scaler from saved json file.

    :param scaler_file:
    :return:
    """
    with open(scaler_file, "r") as file_obj:
        scaler_params = json.load(file_obj)
    scaler = scaler_objs[scaler_params["type"]]()
    del scaler_params["type"]
    for k, v in scaler_params.items():
        if type(v) == list:
            setattr(scaler, k, np.array(scaler_params[k]))
        else:
            setattr(scaler, k, scaler_params[k])
    return scaler