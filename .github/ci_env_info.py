"""Print the parts of the CI environment that decide which bridgescaler code paths the tests exercise."""
import platform
import sys

import numba
import numpy as np

print("python", sys.version.split()[0], platform.platform())
print("numpy", np.__version__, "| numba", numba.__version__)


@numba.njit(parallel=True)
def _row_sum(a):
    out = np.zeros(a.shape[0])
    for i in numba.prange(a.shape[0]):
        out[i] = a[i].sum()
    return out


_row_sum(np.ones((8, 4)))
print("numba threading layer:", numba.threading_layer(), "| threads:", numba.config.NUMBA_NUM_THREADS)

try:
    import torch
except ImportError:
    print("torch not installed")
else:
    print("torch", torch.__version__, "| cuda:", torch.cuda.is_available(), "| mps:", torch.backends.mps.is_available())
