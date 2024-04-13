import matplotlib
matplotlib.use('agg')
import numpy as np
import matplotlib.pyplot as plt
import psutil
import xarray as xr
mem = []
def get_data():
    return np.zeros((1000, 50, 50), dtype=np.float32)
data = get_data()
for i in range(data.shape[0]):
    data[i] = np.random.random((50, 50))
    mem.append(psutil.virtual_memory()[1])
mem.append(psutil.virtual_memory()[1])
xd = xr.DataArray(data)
mem.append(psutil.virtual_memory()[1])
plt.plot(mem)
plt.savefig("mem_profile.png", dpi=200, bbox_inches="tight")
