.. title:: Distributed Scalers

.. distributed:

Distributed Scalers
===================
The distributed scalers allow you to calculate scaling
parameters on different subsets of a dataset and then combine the scaling factors
together to get representative scaling values for the full dataset. Distributed
Standard Scalers, MinMax Scalers, and Quantile Transformers have been implemented and work with both tabular
and muliti-dimensional patch data in numpy, pandas DataFrame, and xarray DataArray formats.

Example:

.. code-block:: python

    from bridgescaler.distributed import DStandardScaler
    import numpy as np

    dss_1 = DStandardScaler()
    dss_2 = DStandardScaler()

    x_1 = np.random.normal(0, 2.2, (20, 5))
    x_2 = np.random.normal(1, 3.5, (25, 5))

    dss_1.fit(x_1)
    dss_2.fit(x_2)
    dss_combined = np.sum([dss_1, dss_2])

    dss_combined.transform(x_1)