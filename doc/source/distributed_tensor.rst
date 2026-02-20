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
