"""Benchmark TDigest fitting: numba per channel, numba prange over channels (CPU), and the batched torch fit
(CPU/CUDA, eager/compiled).

Every torch result is checked bit-for-bit against numba before its time is reported.

    python scripts/benchmark_tdigest_fit.py --channels 1 16 64 --values 1000000 --compression 250
"""
import argparse
import time

import numba
import numpy as np
import torch

from bridgescaler.tdigest import TDigest, TDigestArrays
from bridgescaler.tdigest_tensor import fit_tdigest_batch


def numba_fit(x, compression):
    out = []
    for row in x:
        td = TDigest(compression)
        td.update(row)
        out.append(td)
    return out


def check(batch, digests):
    centroids, size, t_min, t_max = batch.to_states()
    for i, td in enumerate(digests):
        ref = td.centroids()
        got = centroids[i]
        assert got.shape[0] == ref.shape[0], f"channel {i}: centroid count differs"
        assert np.array_equal(got[:, 0].view(np.uint64), ref["mean"].view(np.uint64)), f"channel {i}: means"
        assert np.array_equal(got[:, 1].view(np.uint64), ref["weight"].view(np.uint64)), f"channel {i}: weights"
        assert size[i] == td.size() and t_min[i] == td.min() and t_max[i] == td.max(), f"channel {i}: scalars"


def timed(fn, sync):
    sync()
    t = time.perf_counter()
    out = fn()
    sync()
    return out, time.perf_counter() - t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--channels", type=int, nargs="+", default=[1, 16, 64])
    parser.add_argument("--values", type=int, default=1_000_000)
    parser.add_argument("--compression", type=float, default=250)
    parser.add_argument("--devices", nargs="*", default=["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    numba_fit(rng.normal(size=(1, 1000)), args.compression)  # JIT warm-up
    print(f"values per channel: {args.values}, compression: {args.compression}")
    for n_ch in args.channels:
        x = rng.gamma(2.0, size=(n_ch, args.values))
        digests, t_numba = timed(lambda: numba_fit(x, args.compression), lambda: None)
        print(f"channels={n_ch:4d}  numba (sequential over channels): {t_numba:8.3f} s")
        TDigestArrays.fit(x[:, :1000].T, args.compression)  # JIT warm-up of the parallel kernel
        arrays, t_prange = timed(lambda: TDigestArrays.fit(x.T, args.compression), lambda: None)
        cents = arrays.to_states()[0]
        assert all(np.array_equal(c.view(np.uint64), np.stack([td.centroids()["mean"], td.centroids()["weight"]],
                                                               axis=1).view(np.uint64))
                   for c, td in zip(cents, digests)), "prange result differs"
        print(f"channels={n_ch:4d}  numba prange ({numba.get_num_threads()} threads):    {t_prange:8.3f} s  "
              f"({t_numba / t_prange:5.2f}x numba, 0 ULP verified)")
        for device in args.devices:
            sync = torch.cuda.synchronize if device == "cuda" else (lambda: None)
            xt = torch.from_numpy(x).to(device)
            for compile in ([False] if args.no_compile else [False, True]):
                try:
                    if compile:  # compile/warm up outside the timed region
                        fit_tdigest_batch(xt[:, :20000], args.compression, compile=True)
                    batch, t = timed(lambda: fit_tdigest_batch(xt, args.compression, compile=compile), sync)
                    check(batch, digests)
                    mode = "compiled" if compile else "eager"
                    print(f"channels={n_ch:4d}  torch {device:4s} {mode:8s}: {t:8.3f} s  "
                          f"({t_numba / t:5.2f}x numba, 0 ULP verified)")
                except Exception as e:  # noqa: BLE001 - report and keep benchmarking other modes
                    print(f"channels={n_ch:4d}  torch {device} compile={compile}: failed: {type(e).__name__}: "
                          f"{str(e).splitlines()[0][:200]}")


if __name__ == "__main__":
    main()
