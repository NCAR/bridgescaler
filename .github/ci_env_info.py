"""Print the parts of the CI environment that decide which bridgescaler code paths the tests exercise."""
import ctypes
import platform
import sys

import numba
import numpy as np

import bridgescaler  # noqa: F401  (imports scikit-learn, which may load its own OpenMP runtime)

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

if sys.platform == "darwin":
    # More than one OpenMP runtime in the process is the setup that crashed in issue #44.
    dyld = ctypes.CDLL(None)
    dyld._dyld_get_image_name.restype = ctypes.c_char_p
    images = [dyld._dyld_get_image_name(i).decode() for i in range(dyld._dyld_image_count())]
    omp = [name for name in images if any(k in name.rsplit("/", 1)[-1] for k in ("libomp", "libgomp", "libiomp"))]
    print(f"OpenMP runtimes loaded ({len(omp)}):", *omp, sep="\n  ")

try:
    import torch
except ImportError:
    print("torch not installed")
else:
    print("torch", torch.__version__, "| cuda:", torch.cuda.is_available(), "| mps:", torch.backends.mps.is_available())
