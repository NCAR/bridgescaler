.. title:: Dictionary of Scalers

.. dictionary of scalers:

Dictionary of Scalers
===================
When working with data organized as nested dictionaries, ``scale_var_dict`` provides a convenient way to fit and transform variables while preserving the original structure.

The input to the function is expected to be a dictionary with arbitrary levels of nesting. For example:

.. code-block:: python

    import torch

    T = torch.randn((20, 5, 4, 8))
    var_dict = {
        "era5": {
            "input": {"era5/prognostic/3d/T": T},
            "target": {"era5/prognostic/3d/T": T},
        }
    }

The function always requires an input dictionary, but the expected scaler argument depends on the ``method``.

For ``"fit"`` and ``"fit_transform"``, a scaler object is required:

.. code-block:: python

    from bridgescaler.distributed_tensor import DStandardScalerTensor
    from bridgescaler import scale_var_dict

    scalers = DStandardScalerTensor()  # can be any scaler object
    scaler_dict = scale_var_dict(var_dict, scalers, method="fit")

The returned ``scaler_dict`` is a nested dictionary of scalers that mirrors the structure of the input data.

For ``"transform"`` and ``"inverse_transform"``, the function expects a nested dictionary of scalers and returns data with the same structure as the input:

.. code-block:: python

    transformed = scale_var_dict(var_dict, scaler_dict, method="transform")
    inverse_transformed = scale_var_dict(transformed, scaler_dict, method="inverse_transform")

The fitted scaler dictionary can be saved and reloaded for reuse:

.. code-block:: python

    from bridgescaler import save_scaler_dict, load_scaler_dict
    save_scaler_dict(scaler_dict, 'scaler_dict.json')
    scaler_dict_in = load_scaler_dict('scaler_dict.json')