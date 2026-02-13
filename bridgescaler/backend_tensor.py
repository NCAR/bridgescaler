from . import require_torch
require_torch()   # enforce torch availability/version at import time
import torch

import json
import numpy as np

from bridgescaler.distributed_tensor import DStandardScalerTensor, DMinMaxScalerTensor
from .backend import NumpyEncoder, object_hook


scaler_objs = {"DStandardScalerTensor": DStandardScalerTensor,
               "DMinMaxScalerTensor": DMinMaxScalerTensor,
               }


def print_scaler_tensor(scaler):
    """
    Modify the print_scaler() in backend.py for tensors.
    """
    scaler_params = scaler.__dict__
    scaler_params["type"] = str(type(scaler))[1:-2].split(".")[-1]

    for keys in scaler_params:
        if type(scaler_params[keys]) == torch.Tensor:
            scaler_params[keys] = scaler_params[keys].cpu().numpy().copy()

    return json.dumps(scaler_params, indent=4, sort_keys=True, cls=NumpyEncoder)


def read_scaler_tensor(scaler_str):
    """
    Modify the read_scaler() in backend.py for tensors.
    """
    scaler_params = json.loads(scaler_str, object_hook=object_hook)
    scaler = scaler_objs[scaler_params["type"]]()
    del scaler_params["type"]
    for k, v in scaler_params.items():
        if k == "x_columns_":
            setattr(scaler, k, v)
        else:
            setattr(scaler, k, torch.tensor(v))
    return scaler