.. title:: Distributed Scalers

.. distributed:

Distributed Scalers
===================
The distributed scalers allow you to calculate scaling
parameters on different subsets of a dataset and then combine the scaling factors
together to get representative scaling values for the full dataset. Distributed
Standard Scalers, MinMax Scalers, and Quantile Transformers have been implemented and work with both tabular
and muliti-dimensional patch data in numpy, pandas DataFrame, and xarray DataArray formats.

The distributed scalers allow you to calculate scaling
parameters on different subsets of a dataset and then combine the scaling factors
together to get representative scaling values for the full dataset. Distributed
Standard Scalers, MinMax Scalers, and Quantile Scalers have been implemented and work with both tabular
and muliti-dimensional patch data in numpy, pandas DataFrame, and xarray DataArray formats.
By default, the scaler assumes your channel/variable dimension is the last
dimension, but if `channels_last=False` is set in the `__init__`, `transform`,
or `inverse_transform` methods, then the 2nd dimension is assumed to be the variable
dimension. It is possible to fit data with one ordering and then
transform it with a different one.

For large datasets, it may be expensive to redo the scalers if you want to use a
subset or different ordering of variables. However, in bridgescaler, the
Distributed Scalers all support arbitrary ordering and subsets of variables for transforms if
the input data are in a Xarray DataArray or Pandas DataFrame with variable
names that match the original data.

Example:

.. code-block:: python

    from bridgescaler.distributed import DStandardScaler
    import numpy as np

    x_1 = np.random.normal(0, 2.2, (20, 5, 4, 8))
    x_2 = np.random.normal(1, 3.5, (25, 4, 8, 5))

    dss_1 = DStandardScaler(channels_last=False)
    dss_2 = DStandardScaler(channels_last=True)
    dss_1.fit(x_1)
    dss_2.fit(x_2)
    dss_combined = np.sum([dss_1, dss_2])

    dss_combined.transform(x_1, channels_last=False)

Distributed scalers can be stored in individual json files or within
a pandas DataFrame for easy loading and combining later.

.. code-block:: python

    import pandas as pd
    from bridgescaler import print_scaler, read_scaler
    scaler_list = [dss_1, dss_2]
    df = pd.DataFrame({"scalers": [print_scaler(s) in scaler_list]}])
    df.to_parquet("scalers.parquet")
    df_new = df.read_parquet("scalers.parquet")
    scaler_objs = df_new["scalers"].apply(read_scaler)
    total_scaler = scaler_objs.sum()

