"""Benchmark DQuantileScaler (numpy/crick) vs DQuantileScalerTensor (torch) transform runtimes.

Compares `transform` wall-clock time across total array sizes 1e3 .. 1e6 (powers of 10) for:
  * DQuantileScaler          - numpy/crick, CPU
  * DQuantileScalerTensor    - torch, CPU (eager)
  * DQuantileScalerTensor    - torch, GPU (eager)
  * DQuantileScalerTensor    - torch, GPU (torch.compile)   [if CUDA available]

Fitting is done once per configuration and is NOT timed; only `transform` is timed, over
several repeats after warmup. GPU timings synchronize the device around each call.
"""
import time

import numpy as np
import torch

from bridgescaler.distributed import DQuantileScaler
from bridgescaler.distributed_tensor import DQuantileScalerTensor

N_VARS = 4               # number of variables/channels
SIZES = [1_000, 10_000, 100_000, 1_000_000]   # total array elements
DISTRIBUTION = "normal"
N_REPEAT = 20            # timed repeats per config
N_WARMUP = 3
SEED = 20240706


def make_data(total_size, n_vars=N_VARS):
    """Build a (rows, n_vars) float64 array with total elements ~= total_size."""
    rows = max(total_size // n_vars, 1)
    rng = np.random.default_rng(SEED)
    # give each variable a different distribution so the digests differ
    data = np.empty((rows, n_vars), dtype=np.float64)
    for v in range(n_vars):
        data[:, v] = rng.normal(loc=v * 3.0, scale=1.0 + v, size=rows)
    return data


def time_call(fn, n_repeat=N_REPEAT, n_warmup=N_WARMUP, sync=None):
    """Return (median, min) seconds over n_repeat timed calls after n_warmup warmups."""
    for _ in range(n_warmup):
        fn()
    if sync is not None:
        sync()
    times = []
    for _ in range(n_repeat):
        if sync is not None:
            sync()
        t0 = time.perf_counter()
        fn()
        if sync is not None:
            sync()
        times.append(time.perf_counter() - t0)
    times = np.array(times)
    return float(np.median(times)), float(times.min())


def bench_numpy_cpu(data):
    scaler = DQuantileScaler(distribution=DISTRIBUTION)
    scaler.fit(data)
    ref = scaler.transform(data)
    med, mn = time_call(lambda: scaler.transform(data))
    return med, mn, ref


def bench_tensor(data, device, compile_flag=False, ref=None):
    scaler = DQuantileScalerTensor(distribution=DISTRIBUTION, compile=compile_flag)
    # fit on CPU (fit_variable_tensor pulls data to CPU/numpy internally)
    scaler.fit(torch.from_numpy(data))
    x = torch.from_numpy(data).to(device)
    sync = torch.cuda.synchronize if device.type == "cuda" else None

    out = scaler.transform(x)
    max_err = None
    if ref is not None:
        max_err = float(np.nanmax(np.abs(out.cpu().numpy() - ref)))
    med, mn = time_call(lambda: scaler.transform(x), sync=sync)
    return med, mn, max_err


def main():
    cuda = torch.cuda.is_available()
    print(f"torch {torch.__version__} | CUDA available: {cuda}"
          + (f" | {torch.cuda.get_device_name(0)}" if cuda else ""))
    print(f"n_vars={N_VARS}, distribution={DISTRIBUTION}, "
          f"repeats={N_REPEAT} (median reported), warmup={N_WARMUP}\n")

    header = f"{'size':>10} {'np-cpu (ms)':>13} {'tensor-cpu (ms)':>16} {'tensor-gpu (ms)':>16}"
    if cuda:
        header += f" {'tensor-gpu-compile (ms)':>24}"
    print(header)
    print("-" * len(header))

    results = []
    for size in SIZES:
        data = make_data(size)

        np_med, np_min, ref = bench_numpy_cpu(data)
        tc_med, tc_min, tc_err = bench_tensor(data, torch.device("cpu"), ref=ref)

        row = f"{size:>10} {np_med*1e3:>13.3f} {tc_med*1e3:>16.3f}"
        rec = {"size": size, "np_cpu": np_med, "tensor_cpu": tc_med,
               "tensor_cpu_maxerr": tc_err}

        if cuda:
            dev = torch.device("cuda")
            tg_med, tg_min, tg_err = bench_tensor(data, dev, ref=ref)
            tgc_med, tgc_min, tgc_err = bench_tensor(data, dev, compile_flag=True, ref=ref)
            row += f" {tg_med*1e3:>16.3f} {tgc_med*1e3:>24.3f}"
            rec.update(tensor_gpu=tg_med, tensor_gpu_compile=tgc_med,
                       tensor_gpu_maxerr=tg_err)
        else:
            row += f" {'n/a':>16}"
        print(row)
        results.append(rec)

    # correctness summary
    print("\nMax abs error vs numpy transform (should be ~1e-6 or smaller):")
    for rec in results:
        msg = f"  size={rec['size']:>9}: cpu={rec['tensor_cpu_maxerr']:.2e}"
        if "tensor_gpu_maxerr" in rec:
            msg += f", gpu={rec['tensor_gpu_maxerr']:.2e}"
        print(msg)


if __name__ == "__main__":
    main()
