import numpy as np
import matplotlib.pyplot as plt
import psutil
import xarray as xr
mem = []
mem.append(psutil.virtual_memory()[3])
def get_data():
    return np.zeros((1000, 50, 50), dtype=np.float32)
data = get_data()
mem.append(psutil.virtual_memory()[3])
for i in range(data.shape[0]):
    data[i] = np.random.random((50, 50))
    mem.append(psutil.virtual_memory()[3])
mem.append(psutil.virtual_memory()[3])
xd = xr.DataArray(data)
mem.append(psutil.virtual_memory()[3])
plt.plot(mem)
plt.show()
