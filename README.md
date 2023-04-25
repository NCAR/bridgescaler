# bridgescaler
Bridge your scikit-learn scaler parameters between Python sessions and users.
Bridgescaler allows you to save the properties of a scikit-learn scaler object
to a json file, and then repopulate a new scaler object with the same properties.


## Dependencies
* scikit-learn
* numpy
* pandas

## Installation
For a stable version of bridgescaler, you can install from PyPI.
```bash
pip install bridgescaler
```

For the latest version of bridgescaler, install from github.
```bash
git clone https://github.com/NCAR/bridgescaler.git
cd bridgescaler
pip install .
```

## Usage
bridgescaler supports all the common scikit-learn scaler classes:
* StandardScaler
* RobustScaler
* MinMaxScaler
* MaxAbsScaler
* QuantileTransformer
* PowerTransformer
* SplineTransformer

First, create some synthetic data to transform.
```python
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
```

Now, let's fit and transform the data with StandardScaler.
```python
from sklearn.preprocessing import StandardScaler
from bridgescaler import save_scaler, load_scaler
scaler = StandardScaler()
scaler.fit_transform(x_data)
filename = "x_standard_scaler.json"
# save to json file
save_scaler(scaler, filename)

# create new StandardScaler from json file information.
new_scaler = load_scaler(filename) # new_scaler is a StandardScaler object
```

### Group Scaler
The group scalers use the same scaling parameters for a group of similar
variables rather than scaling each column independently. This is useful for situations where variables are related, 
such as temperatures at different height levels.

Groups are specified as a list of column ids, which can be column names for pandas dataframes or column indices
for numpy arrays.

For example:
```python
from bridgescaler.group import GroupStandardScaler
import pandas as pd
import numpy as np
x_rand = np.random.random(size=(100, 5))
data = pd.DataFrame(data=x_rand, 
                    columns=["a", "b", "c", "d", "e"])
groups = [["a", "b"], ["c", "d"], "e"]
group_scaler = GroupStandardScaler()
x_transformed = group_scaler.fit_transform(data, groups=groups)
```

"a" and "b" are a single group and all values of both will be included when calculating the mean and standard 
deviation for that group.

### Deep Scaler
The deep scalers are designed to scale 2 or 3 dimensional fields input into a 
deep learning model such as a convolutional neural network. The scalers assume
that the last dimension is the channel/variable dimension and scales the values accordingly.
The scalers can support 2D or 3D patches with no change in code structure.

Example:
```python
from bridgescaler.deep import DeepStandardScaler
import numpy as np
np.random.seed(352680)
n_ex = 5000
n_channels = 4
dim = 32
means = np.array([1, 5, -4, 2.5], dtype=np.float32)
sds = np.array([10, 2, 43.4, 32.], dtype=np.float32)
x = np.zeros((n_ex, dim, dim, n_channels), dtype=np.float32)
for chan in range(n_channels):
    x[..., chan] = np.random.normal(means[chan], sds[chan], (n_ex, dim, dim))
dss = DeepStandardScaler()
dss.fit(x)
x_transformed = dss.transform(x)
```
