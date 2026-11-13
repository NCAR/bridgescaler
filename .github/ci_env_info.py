"""Print the parts of the CI environment that decide which bridgescaler code paths the tests exercise."""
import ctypes
import platform
import subprocess
import sys
import textwrap

import numba
import numpy as np

import bridgescaler  # noqa: F401  (imports scikit-learn, which may load its own OpenMP runtime)
from bridgescaler.tdigest import PARALLEL_MODE

print("python", sys.version.split()[0], platform.platform())
print("numpy", np.__version__, "| numba", numba.__version__)
print("bridgescaler.tdigest.PARALLEL_MODE:", PARALLEL_MODE)

# A trivial numba prange loop after importing bridgescaler, in a child process: in the environment from issue #44
# (two OpenMP runtimes loaded) numba's omp layer segfaults here, and that should be reported, not kill this script.
PROBE = textwrap.dedent("""
    import ctypes, sys
    import numba, numpy as np
    import bridgescaler

    @numba.njit(parallel=True)
    def row_sum(a):
        out = np.zeros(a.shape[0])
        for i in numba.prange(a.shape[0]):
            out[i] = a[i].sum()
        return out

    row_sum(np.ones((8, 4)))
    print("numba threading layer:", numba.threading_layer(), "| threads:", numba.config.NUMBA_NUM_THREADS)
    if sys.platform == "darwin":
        dyld = ctypes.CDLL(None)
        dyld._dyld_get_image_name.restype = ctypes.c_char_p
        images = [dyld._dyld_get_image_name(i).decode() for i in range(dyld._dyld_image_count())]
        omp = [n for n in images if any(k in n.rsplit("/", 1)[-1] for k in ("libomp", "libgomp", "libiomp"))]
        print(f"OpenMP runtimes loaded ({len(omp)}):", *omp, sep="\\n  ")
""")
probe = subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True)
print(probe.stdout.rstrip())
if probe.returncode != 0:
    print(f"numba prange probe FAILED with exit code {probe.returncode}:\n{probe.stderr.rstrip()}")

try:
    import torch
except ImportError:
    print("torch not installed")
else:
    print("torch", torch.__version__, "| cuda:", torch.cuda.is_available(), "| mps:", torch.backends.mps.is_available())
