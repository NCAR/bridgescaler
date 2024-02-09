.. title:: Group Scalers

.. group:

Group Scalers
=============

The group scalers use the same scaling parameters for a group of similar
variables rather than scaling each column independently. This is useful for situations where variables are related,
such as temperatures at different height levels.

Groups are specified as a list of column ids, which can be column names for pandas dataframes or column indices
for numpy arrays.

For example:

.. code-block:: python

    from bridgescaler.group import GroupStandardScaler
    import pandas as pd
    import numpy as np
    x_rand = np.random.random(size=(100, 5))
    data = pd.DataFrame(data=x_rand,
                        columns=["a", "b", "c", "d", "e"])
    groups = [["a", "b"], ["c", "d"], "e"]
    group_scaler = GroupStandardScaler()
    x_transformed = group_scaler.fit_transform(data, groups=groups)

"a" and "b" are a single group and all values of both will be included when calculating the mean and standard
deviation for that group.