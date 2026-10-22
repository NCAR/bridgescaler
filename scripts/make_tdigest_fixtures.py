"""Generate the crick reference fixture used by bridgescaler/tests/tdigest_test.py.

Needs an environment with crick installed (``pip install bridgescaler[test-crick]``). Run from the repo root:

    python scripts/make_tdigest_fixtures.py
"""
import os
import sys

import numpy as np
import crick
from crick import TDigest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "bridgescaler", "tests"))
import tdigest_cases as tc  # noqa: E402

OUT = os.path.join(REPO, "bridgescaler", "tests", "data", "tdigest_reference.npz")


def main():
    out = {"crick_version": np.array(crick.__version__), "numpy_version": np.array(np.__version__)}
    data_cache = {}
    for name, c, variant in tc.iter_cases():
        if name not in data_cache:
            # store the inputs: numpy's normal/gamma/... samplers are not bit-identical across platforms
            data_cache[name] = tc.make_data(name)
            out[f"{name}|data"] = data_cache[name]
        centroids, size, t_min, t_max = tc.run_case(TDigest, name, c, variant, data_cache[name])
        key = tc.case_key(name, c, variant)
        out[key + "|centroids"] = centroids
        out[key + "|scalars"] = np.array([size, t_min, t_max], dtype=np.float64)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(OUT, **out)
    print(f"Wrote {OUT} ({os.path.getsize(OUT) / 1e6:.2f} MB) with crick {crick.__version__}")


if __name__ == "__main__":
    main()
