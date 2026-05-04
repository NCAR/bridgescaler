import copy
import importlib

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


def ensure_torch():
    """
    Validates torch installation and load the module.
    """
    from . import require_torch
    require_torch()  # enforce torch availability/version at import time
    import torch
    return torch


def save_scaler(scaler, scaler_file):
    """
    Save a scikit-learn or bridgescaler scaler object to json format.

    Args:
        scaler: scikit-learn-style scaler object
        scaler_file: path to json file where scaler information is stored.
    """
    scaler = copy.deepcopy(scaler)

    scaler_params = scaler.__dict__
    scaler_params["type"] = str(type(scaler))[1:-2].split(".")[-1]

    with open(scaler_file, "w") as file_obj:
        if "Tensor" in scaler_params["type"]:
            torch = ensure_torch()

            for keys in scaler_params:
                if type(scaler_params[keys]) == torch.Tensor:
                    scaler_params[keys] = scaler_params[keys].cpu().numpy()
                elif (keys == "centroids_mean_tensor") or (keys == "centroids_weight_tensor"):
                    scaler_params[keys] = [c.cpu().numpy() for c in scaler_params[keys]]
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
    scaler = copy.deepcopy(scaler)

    scaler_params = scaler.__dict__
    scaler_params["type"] = str(type(scaler))[1:-2].split(".")[-1]

    if "Tensor" in scaler_params["type"]:
        torch = ensure_torch()

        for keys in scaler_params:
            if type(scaler_params[keys]) == torch.Tensor:
                scaler_params[keys] = scaler_params[keys].cpu().numpy()
            elif (keys == "centroids_mean_tensor") or (keys == "centroids_weight_tensor"):
                scaler_params[keys] = [c.cpu().numpy() for c in scaler_params[keys]]
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

    is_tensor = "Tensor" in scaler_params["type"]

    if is_tensor:
        torch = ensure_torch()

        scaler_class = getattr(importlib.import_module("bridgescaler.distributed_tensor"), scaler_params["type"])
        scaler = scaler_class()
    else:
        scaler = scaler_objs[scaler_params["type"]]()
    del scaler_params["type"]

    for k, v in scaler_params.items():
        # 1. Handle Serialized Numpy Arrays
        if isinstance(v, dict) and v.get("object") == "ndarray":
            value = np.array(v['data'], dtype=v['dtype']).reshape(v['shape'])

        # 2. Handle Tensors & Special Cases
        elif is_tensor:
            if isinstance(v, str) or (k == "x_columns_") or (k == "centroids_"):
                value = v # keep as it is
            elif (k == "centroids_mean_tensor") or (k == "centroids_weight_tensor"):
                value = [torch.tensor(c) for c in v]  # convert to a list with tensors
            elif isinstance(v, np.bool_):
                value = torch.tensor(bool(v))
            else:
                value = torch.tensor(v)

        # 3. Fallback for primitives
        else:
            value = v

        setattr(scaler, k, value)
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


def apply_to_dict_leaves(d, operation):
    """
    Recursively applies an operation to each leaf value in a nested dictionary.

    Args:
        d (dict): A nested dictionary where the operation will be
            applied to each leaf value.
        operation (callable): A function to apply to each leaf value.

    Returns:
        dict: A nested dictionary with the same structure as ``d``,
            where each leaf is the result of ``operation(leaf)``.
    """
    result = {}
    for key, value in d.items():
        if isinstance(value, dict):
            result[key] = apply_to_dict_leaves(value, operation)
        else:
            result[key] = operation(value)
    return result


def save_scaler_dict(scaler_dict, scaler_dict_file):
    """
    Serializes and saves a nested dictionary of Bridgescaler scalers to a JSON file.

    Args:
        scaler_dict (dict): A nested dictionary of fitted Bridgescaler scaler objects
            to be saved.
        scaler_dict_file (str or Path): The file path where the scaler
            dictionary will be saved as a JSON file.
    """
    with open(scaler_dict_file, "w") as file_obj:
        json.dump(apply_to_dict_leaves(scaler_dict, print_scaler), file_obj, indent=4, sort_keys=True, cls=NumpyEncoder)


def load_scaler_dict(scaler_dict_file):
    """
    Loads and deserializes a nested dictionary of Bridgescaler scalers from a JSON file.

    Args:
        scaler_dict_file (str or Path): The file path to the JSON file
            containing the serialized scaler dictionary.

    Returns:
        dict: A nested dictionary of reconstructed scaler objects, with the
            same structure as the original dictionary passed to
            ``save_scaler_dict``.
    """
    with open(scaler_dict_file, "r") as file_obj:
        scaler_str = json.load(file_obj)
    return apply_to_dict_leaves(scaler_str, read_scaler)


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

        elif isinstance(obj, (np.complex64, np.complex128)):
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

        elif isinstance(obj, pd.api.extensions.ExtensionArray):
            return obj.tolist()

        return json.JSONEncoder.default(self, obj)


def scale_var_dict(var_dict, scalers, method, var_list=None, _key_path=()):
    """
    Recursively traverses a nested dict of tensor variables and applies a scaler method to each variable.

    Args:
        var_dict (dict): A nested dictionary where leaves are variables in torch.Tensor to be scaled.
        scalers (object or dict): A single scaler instance (for ``fit`` and
            ``fit_transform``) or a nested dict of scalers matching the structure
            of ``var_dict`` (for ``transform`` and ``inverse_transform``).
        method (str): The scaler method to apply. Must be one of ``fit``,
            ``transform``, ``inverse_transform``, or ``fit_transform``.
        var_list (list of str, optional): A list of leaf key names to apply the
            scaler method to. Keys not in ``var_list`` are skipped during ``fit``,
            and left unchanged during ``transform``, ``inverse_transform``, and
            ``fit_transform``. If ``None``, all leaf keys are processed.

    Returns:
        dict: A nested dictionary with the same structure as ``var_dict``,
            where each leaf is either a fitted scaler (for ``fit``) or a
            transformed variable (for ``transform``, ``inverse_transform``,
            ``fit_transform``). Keys named ``metadata`` and keys excluded by
            ``var_list`` are omitted for ``fit``, and passed through unchanged
            for other methods.

    Raises:
        AssertionError: If ``var_dict`` is not a dict.
        AssertionError: If ``method`` is not one of the valid methods.
        AssertionError: If ``scalers`` is not a dict when using ``transform``
            or ``inverse_transform``.
        AssertionError: If a key path in ``var_dict`` is missing in ``scalers``.
        AssertionError: If a scaler at a given key path does not have the
            requested ``method``.

    Example:
        >>> import torch
        >>> from bridgescaler.distributed_tensor import DStandardScalerTensor
        >>> from bridgescaler.backend import scale_var_dict
        >>> T = torch.randn((20, 5, 4, 8))
        >>> var_dict = {
            "era5": {
                "input": {"era5/prognostic/3d/T": T},
                "target": {"era5/prognostic/3d/T": T},
                "metadata": {"input_datetime": int, "target_datetime": int}
                }
            }
        >>> scalers = DStandardScaler(channels_last=False)
        >>> scaler_dict = scale_var_dict(var_dict, scalers, method="fit")
        >>> transformed = scale_var_dict(var_dict, scaler_dict, method="transform")
        >>> inverse_transformed = scale_var_dict(transformed, scaler_dict, method="inverse_transform")
        >>> fitted_transformed = scale_var_dict(var_dict, scalers, method="fit_transform")
        >>> # Only scale specific variables
        >>> filtered = scale_var_dict(var_dict, scaler_dict, method="transform", var_list=["era5/prognostic/3d/T"])
    """
    VALID_METHODS = {"fit", "transform", "inverse_transform", "fit_transform"}
    is_fit = "fit" in method

    # Validate top-level inputs
    assert isinstance(var_dict, dict), f"Expected 'var_dict' to be a dict, got {type(var_dict).__name__}"
    assert method in VALID_METHODS, f"Invalid method '{method}'. Choose from {VALID_METHODS}"
    assert isinstance(scalers, dict) or hasattr(scalers, method), (
        f"'scalers' must be a dict or a scaler object with a '{method}' method"
    )
    if not is_fit:
        assert isinstance(scalers, dict), (
            f"For method '{method}', 'scalers' must be a dict matching the structure of 'var_dict'"
        )

    result = {}
    for key, value in var_dict.items():
        current_path = _key_path + (key,)
        path_str = " -> ".join(str(k) for k in current_path)

        if key == "metadata":
            if method != "fit":
                result[key] = value
            continue

        is_leaf = not isinstance(value, dict)
        is_excluded = var_list is not None and current_path[-1] not in var_list
        if is_leaf and is_excluded:
            if method != "fit":
                result[key] = value
            continue

        if not is_fit:
            assert key in scalers, (
                f"Key path '{path_str}' found in 'var_dict' but missing in 'scalers'"
            )

        scaler = scalers if is_fit else scalers[key]

        if isinstance(value, dict):
            result[key] = scale_var_dict(value, scaler, method, var_list, current_path)
        else:
            scaler = copy.deepcopy(scaler)
            assert hasattr(scaler, method), (
                f"Scaler at key path '{path_str}' does not have a '{method}' method, got {type(scaler).__name__}"
            )
            result[key] = getattr(scaler, method)(value)
            if method == "fit":
                result[key] = scaler

    return result


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