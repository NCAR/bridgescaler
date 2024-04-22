from scipy.stats import norm
from scipy.special import ndtri
import numpy as np
import matplotlib.pyplot as plt
import psutil
import gc

process = psutil.Process()
n_elements = 301
mem_vals = np.zeros(n_elements)
mem_vals[0] = process.memory_info().rss / 1e6
for i in range(1, n_elements):
    x = np.random.random(size=(100, 50, 50))
    ppf_val = ndtri(x)
    mem_vals[i] = process.memory_info().rss / 1e6
    gc.collect()
plt.plot(mem_vals[1:] - mem_vals[0], label="ndtri")
mem_vals = np.zeros(n_elements)
mem_vals[0] = process.memory_info().rss / 1e6

for i in range(1, n_elements):
    x = np.random.random(size=(100, 50, 50))
    ppf_val = norm.ppf(x)
    mem_vals[i] = process.memory_info().rss / 1e6
    gc.collect()
plt.plot(mem_vals[1:] - mem_vals[0], label="norm.ppf")
mem_vals = np.zeros(n_elements)
mem_vals[0] = process.memory_info().rss / 1e6
for i in range(1, n_elements):
    x = np.random.random(size=(100, 50, 50))
    mem_vals[i] = process.memory_info().rss / 1e6
    gc.collect()
plt.plot(mem_vals[1:] - mem_vals[0], label="control")
plt.xlabel("Iterations")
plt.ylabel("Memory usage (MB)")
plt.legend()
plt.savefig("norm_usage_tracking.png", dpi=200, bbox_inches="tight")
