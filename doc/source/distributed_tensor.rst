.. title:: Distributed Scalers for PyTorch tensor

.. distributed for PyTorch tensor:

Distributed Scalers for PyTorch tensor
===================
The distributed scalers for PyTorch tensors behave similarly to standard distributed scalers, but operate directly on ``torch.Tensor`` objects and support optional metadata.

Users may assign variable/channel names to a tensor by setting the ``variable_names`` attribute:

.. code-block:: python

    tensor.variable_names = ['ch1', 'ch2', 'ch3', 'ch4', 'ch5']

Requirements:

- ``variable_names`` must be a list of unique strings
- The number of names must match the size of the tensor's channel dimension
- The attribute should be assigned after moving the tensor to its target device

If ``variable_names`` is not provided, channel alignment checks is limited to total counts and the order cannot be validated.

Example:

.. code-block:: python

    from bridgescaler.distributed_tensor import DStandardScalerTensor
    import numpy as np

    x_1 = torch.from_numpy(np.random.normal(0, 2.2, (20, 5, 4, 8))).to("cuda:0")
    x_2 = torch.from_numpy(np.random.normal(1, 3.5, (25, 4, 8, 5))).to("cuda:0")

    x_1.variable_names = ["ch1", "ch2", "ch3", "ch4", "ch5"]
    x_2.variable_names = ["ch1", "ch2", "ch3", "ch4", "ch5"]

    dss_1 = DStandardScalerTensor(channels_last=False)
    dss_2 = DStandardScalerTensor(channels_last=True)
    dss_1.fit(x_1)
    dss_2.fit(x_2)
    dss_combined = dss_1 + dss_2

    dss_combined.transform(x_1, channels_last=False)

Functions in ``bridgescaler.backend`` module all support distributed scalers for PyTorch tensor.

Per-location standardization
----------------------------
``DStandardSpatialScalerTensor`` keeps a separate mean and variance for every channel *and* every
grid point along ``spatial_dims`` (default ``(-2, -1)``, e.g. latitude x longitude; use
``(-3, -2, -1)`` for level x latitude x longitude). All other axes (batch, time, ...) are reduced.
The statistics ``mean_x_``, ``var_x_`` and ``n_`` have shape ``(C, *spatial_shape_)``.

- NaNs are ignored per location, so missing values at fixed positions are counted correctly. A
  location that is NaN in every sample has NaN statistics and transforms to NaN.
- Locations with zero variance use a standard deviation of 1, so ``transform`` returns ``x - mean``.
- Scalers fit on separate chunks combine with ``+`` (or repeated ``fit`` calls), as with the other
  distributed scalers. All chunks must share the same spatial shape.
- ``spatial_dims`` refer to the scaler's own ``channels_last`` layout. When ``transform`` is called
  with a ``channels_last`` override, the spatial axes keep the same meaning.

Example (CREDIT-style ``(batch, variable, time, lat, lon)`` tensors):

.. code-block:: python

    from bridgescaler import DStandardSpatialScalerTensor, save_scaler, load_scaler
    import torch

    x_1 = torch.randn(8, 1, 2, 192, 288) * 1000 + 98000
    x_2 = torch.randn(4, 1, 2, 192, 288) * 1000 + 98000
    x_1.variable_names = ["PS"]
    x_2.variable_names = ["PS"]

    dss_1 = DStandardSpatialScalerTensor(channels_last=False, spatial_dims=(-2, -1))
    dss_2 = DStandardSpatialScalerTensor(channels_last=False, spatial_dims=(-2, -1))
    dss_1.fit(x_1)
    dss_2.fit(x_2)
    dss_combined = dss_1 + dss_2      # mean_x_.shape == (1, 192, 288)

    x_norm = dss_combined.transform(x_1)
    save_scaler(dss_combined, "ps_spatial_scaler.json")
    dss_loaded = load_scaler("ps_spatial_scaler.json")
