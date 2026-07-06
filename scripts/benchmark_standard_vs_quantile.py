"""Compare DStandardScalerTensor vs DQuantileScalerTensor transform time at 1M elements.

Configurations: CPU (eager), GPU (eager), GPU (torch.compile). DQuantileScalerTensor has a
built-in `compile` flag; DStandardScalerTensor does not, so its transform is wrapped in
torch.compile manually for the compiled case. Only `transform` is timed (fit excluded),
median of N_REPEAT repeats after warmup, with CUDA synchronization around GPU calls.
"""
import time

import numpy as np
import torch

from bridgescaler.distributed_tensor import DStandardScalerTensor, DQuantileScalerTensor

N_VARS = 4
TOTAL_SIZE = 1_000_000
DISTRIBUTION = "normal"
N_REPEAT = 20
N_WARMUP = 5
SEED = 20240706


def make_data():
    rows = TOTAL_SIZE // N_VARS
    rng = np.random.default_rng(SEED)
    data = np.empty((rows, N_VARS), dtype=np.float64)
    for v in range(N_VARS):
        data[:, v] = rng.normal(v * 3.0, 1.0 + v, rows)
    return data


def time_call(fn, sync=None):
    for _ in range(N_WARMUP):
        fn()
    if sync is not None:
        sync()
    times = []
    for _ in range(N_REPEAT):
        if sync is not None:
            sync()
        t0 = time.perf_counter()
        fn()
        if sync is not None:
            sync()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def main():
    cuda = torch.cuda.is_available()
    print(f"torch {torch.__version__} | CUDA: {cuda}"
          + (f" | {torch.cuda.get_device_name(0)}" if cuda else ""))
    print(f"size={TOTAL_SIZE}, n_vars={N_VARS}, distribution={DISTRIBUTION}, "
          f"repeats={N_REPEAT} (median ms)\n")

    data = make_data()

    def std_scaler():
        s = DStandardScalerTensor()
        s.fit(torch.from_numpy(data))
        return s

    def quant_scaler(compile_flag=False):
        s = DQuantileScalerTensor(distribution=DISTRIBUTION, compile=compile_flag)
        s.fit(torch.from_numpy(data))
        return s

    configs = []  # (label, seconds)

    # ---- CPU eager ----
    dev = torch.device("cpu")
    x = torch.from_numpy(data).to(dev)
    s = std_scaler()
    configs.append(("DStandardScalerTensor  CPU (eager)", time_call(lambda: s.transform(x))))
    q = quant_scaler()
    configs.append(("DQuantileScalerTensor  CPU (eager)", time_call(lambda: q.transform(x))))

    if cuda:
        dev = torch.device("cuda")
        x = torch.from_numpy(data).to(dev)
        sync = torch.cuda.synchronize

        # ---- GPU eager ----
        s = std_scaler()
        configs.append(("DStandardScalerTensor  GPU (eager)", time_call(lambda: s.transform(x), sync=sync)))
        q = quant_scaler()
        configs.append(("DQuantileScalerTensor  GPU (eager)", time_call(lambda: q.transform(x), sync=sync)))

        # ---- GPU compiled ----
        # DStandardScalerTensor has no built-in compile flag, and torch.compile(s.transform) drags the
        # CPU-resident stat tensors + device-moves into the graph (forcing a CPU C++ compile). So we
        # pre-move the fitted stats to GPU and compile the pure elementwise standardization, keeping the
        # graph all-CUDA (Triton) -- the fair analogue of DQuantileScalerTensor's built-in GPU compile.
        s = std_scaler()
        g_mean = s.mean_x_.to(dev)
        g_var = s.var_x_.to(dev)

        def std_math(xt):
            return (xt - g_mean) / torch.sqrt(g_var)  # channels_last broadcast over last dim

        s_compiled = torch.compile(std_math, fullgraph=True)
        configs.append(("DStandardScalerTensor  GPU (compile)", time_call(lambda: s_compiled(x), sync=sync)))
        q = quant_scaler(compile_flag=True)
        configs.append(("DQuantileScalerTensor  GPU (compile)", time_call(lambda: q.transform(x), sync=sync)))

    width = max(len(lbl) for lbl, _ in configs)
    print(f"{'config'.ljust(width)}   median (ms)")
    print("-" * (width + 15))
    for lbl, sec in configs:
        print(f"{lbl.ljust(width)}   {sec*1e3:>9.3f}")


if __name__ == "__main__":
    main()
