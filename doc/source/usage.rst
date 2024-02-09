.. title:: Basic Usage

.. usage:

Basic Usage
===========
bridgescaler supports all the common scikit-learn scaler classes:

* StandardScaler
* RobustScaler
* MinMaxScaler
* MaxAbsScaler
* QuantileTransformer
* PowerTransformer
* SplineTransformer

First, create some synthetic data to transform.

.. code-block:: python

    import numpy as np
    import pandas as pd

    # specify distribution parameters for each variable
    locs = np.array([0, 5, -2, 350.5], dtype=np.float32)
    scales = np.array([1.0, 10, 0.1, 5000.0])
    names = ["A", "B", "C", "D"]
    num_examples = 205
    x_data_dict = {}
    for l in range(locs.shape[0]):
        # sample from random normal with different parameters
        x_data_dict[names[l]] = np.random.normal(loc=locs[l], scale=scales[l], size=num_examples)
    x_data = pd.DataFrame(x_data_dict)

Now, let's fit and transform the data with StandardScaler.

.. code-block:: python

    from sklearn.preprocessing import StandardScaler
    from bridgescaler import save_scaler, load_scaler
    scaler = StandardScaler()
    scaler.fit_transform(x_data)
    filename = "x_standard_scaler.json"
    # save to json file
    save_scaler(scaler, filename)
    # create new StandardScaler from json file information.
    new_scaler = load_scaler(filename)
    # new_scaler is a StandardScaler object