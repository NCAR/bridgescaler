from sklearn.preprocessing import (StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, QuantileTransformer,
                                   SplineTransformer, PowerTransformer)
from bridgescaler.group import GroupStandardScaler, GroupRobustScaler, GroupMinMaxScaler
from bridgescaler.deep import DeepStandardScaler, DeepMinMaxScaler, DeepQuantileTransformer
from bridgescaler.distributed import DStandardScaler, DMinMaxScaler, DQuantileScaler
import numpy as np
import json
import pandas as pd
from numpy.lib.format import descr_to_dtype, dtype_to_descr
from base64 import b64decode, b64encode
from typing import Any


scaler_objs = {"StandardScaler": StandardScaler,
               "MinMaxScaler": MinMaxScaler,
               "RobustScaler": RobustScaler,
               "MaxAbsScaler": MaxAbsScaler,
               "SplineTransformer": SplineTransformer,
               "PowerTransformer": PowerTransformer,
               "QuantileTransformer": QuantileTransformer,
               "GroupStandardScaler": GroupStandardScaler,
               "GroupRobustScaler": GroupRobustScaler,
               "GroupMinMaxScaler": GroupMinMaxScaler,
               "DeepStandardScaler": DeepStandardScaler,
               "DeepMinMaxScaler": DeepMinMaxScaler,
               "DeepQuantileTransformer": DeepQuantileTransformer,
               "DStandardScaler": DStandardScaler,
               "DMinMaxScaler": DMinMaxScaler,
               "DQuantileScaler": DQuantileScaler,
               }


def save_scaler(scaler, scaler_file):
    """
    Save a scikit-learn or bridgescaler scaler object to json format.

    Args:
        scaler: scikit-learn-style scaler object
        scaler_file: path to json file where scaler information is stored.
    """
    scaler_params = scaler.__dict__
    scaler_params["type"] = str(type(scaler))[1:-2].split(".")[-1]
    with open(scaler_file, "w") as file_obj:
        json.dump(scaler_params, file_obj, indent=4, sort_keys=True, cls=NumpyEncoder)
    return


def print_scaler(scaler):
    """
    Output scikit-learn or bridgescaler scaler object to json string.

    Args:
        scaler: scikit-learn-style scaler object

    Returns:
        str representation of object in json format
    """
    scaler_params = scaler.__dict__
    scaler_params["type"] = str(type(scaler))[1:-2].split(".")[-1]
    return json.dumps(scaler_params, indent=4, sort_keys=True, cls=NumpyEncoder)


def object_hook(dct: dict[Any, Any]):
    if "__numpy__" in dct:
        np_obj = np.frombuffer(
            b64decode(dct["__numpy__"]), descr_to_dtype(dct["dtype"])
        )
        return np_obj.reshape(shape) if (shape := dct["shape"]) else np_obj[0]
    return dct

def read_scaler(scaler_str):
    """
    Initialize scikit-learn or bridgescaler scaler from json str.

    Args:
        scaler_str: json str

    Returns:
        scaler object.
    """
    scaler_params = json.loads(scaler_str, object_hook=object_hook)
    scaler = scaler_objs[scaler_params["type"]]()
    del scaler_params["type"]
    for k, v in scaler_params.items():
        if isinstance(v, dict) and v["object"] == "ndarray":
            setattr(scaler, k, np.array(v['data'], dtype=v['dtype']).reshape(v['shape']))
        else:
            setattr(scaler, k, v)
    return scaler


def load_scaler(scaler_file):
    """
    Initialize scikit-learn or bridgescaler scaler from saved json file.

    Args:
        scaler_file: path to json file.

    Returns:
        scaler object.
    """
    with open(scaler_file, "r") as file_obj:
        scaler_str = file_obj.read()
    return read_scaler(scaler_str)


class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """

    def default(self, obj):
        if int(np.__version__.split('.')[0]) >= 2:
            float_types = (np.float16, np.float32, np.float64)
        else:
            float_types = (np.float_, np.float16, np.float32, np.float64)
      
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):

            return int(obj)

        elif isinstance(obj, float_types):
            return float(obj)

        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}

        elif isinstance(obj, (np.ndarray,)) and obj.dtype == "|O":
            return {'object': 'ndarray', 'dtype': obj.dtype.str, 'shape': list(obj.shape),
                    'data': obj.ravel().tolist()}

        elif isinstance(obj, (np.ndarray, np.generic)):
            return {
                "__numpy__": b64encode(
                    obj.data if obj.flags.c_contiguous else obj.tobytes()
                ).decode(),
                "dtype": dtype_to_descr(obj.dtype),
                "shape": obj.shape,
            }

        elif isinstance(obj, (np.bool_)):
            return bool(obj)

        elif isinstance(obj, (np.void)):
            return None

        return json.JSONEncoder.default(self, obj)


def create_synthetic_data():
    locs = np.array([0, 5, -2, 350.5], dtype=np.float32)
    scales = np.array([1.0, 10, 0.1, 5000.0])
    names = ["A", "B", "C", "D"]
    num_examples = 205
    x_data_dict = {}
    for l in range(locs.shape[0]):
        x_data_dict[names[l]] = np.random.normal(loc=locs[l], scale=scales[l], size=num_examples)
    x_data = pd.DataFrame(x_data_dict)
    return x_data
