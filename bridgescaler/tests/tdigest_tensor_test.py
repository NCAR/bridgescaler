import functools
import os

import numpy as np
import pytest
import torch
from numba import njit

import tdigest_cases as tc
from bridgescaler.tdigest import TDigest, _asin
from bridgescaler.tdigest_tensor import (TDigestBatch, asin_tensor, fit_tdigest_batch, merge_tdigest_batch,
                                         COMPILE_OPTIONS)

FIXTURE = os.path.join(os.path.dirname(__file__), "data", "tdigest_reference.npz")
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
N_SHORT = 6000


@functools.lru_cache(maxsize=None)
def _inductor_available(device):
    """torch.compile needs a working C++ (CPU) or Triton (CUDA) toolchain; skip compiled tests without one."""
    try:
        fn = torch.compile(lambda t: t * 2 + 1, fullgraph=True, options=COMPILE_OPTIONS)
        fn(torch.randn(4, dtype=torch.float64, device=device))
    except Exception:
        return False
    return True


def _modes():
    for device in DEVICES:
        yield pytest.param(device, False, id=f"{device}-eager")
        yield pytest.param(device, True, id=f"{device}-compile")


def _skip_compile(device, compile):
    # Compiling the flush kernels takes minutes per shape, so compiled tests are opt-in.
    if compile and os.environ.get("BRIDGESCALER_TEST_COMPILE") != "1":
        pytest.skip("compiled TDigest tests are opt-in: set BRIDGESCALER_TEST_COMPILE=1")
    if compile and not _inductor_available(device):
        pytest.skip(f"torch.compile toolchain unavailable on {device}")


def bits(a):
    return np.ascontiguousarray(np.asarray(a, dtype=np.float64)).view(np.uint64)


def assert_state_equal(state, ref, label):
    centroids, size, t_min, t_max = state
    assert centroids.shape == ref[0].shape, f"{label}: centroid count {centroids.shape} != {ref[0].shape}"
    n_diff = int(np.count_nonzero(bits(centroids) != bits(ref[0])))
    assert n_diff == 0, f"{label}: {n_diff} centroid values differ"
    assert np.array_equal(bits([size, t_min, t_max]), bits(ref[1:])), f"{label}: size/min/max differ"


def batch_states(batch):
    centroids, size, t_min, t_max = batch.to_states()
    return [(centroids[i], size[i], t_min[i], t_max[i]) for i in range(len(centroids))]


def short_matrix(n=N_SHORT):
    """Every fixed-length dataset (21 distributions minus the tiny ones) as rows, truncated to n values."""
    names = [name for name in tc.DATASETS if tc.make_data(name).size == tc.N]
    return names, np.stack([tc.make_data(name)[:n].astype(np.float64) for name in names])


@njit
def _asin_many(x, out):
    for i in range(x.size):
        out[i] = _asin(x[i])


@pytest.mark.parametrize("device", DEVICES)
def test_asin_tensor_matches_numba(device):
    rng = np.random.default_rng(3)
    x = np.concatenate([2 * rng.random(2_000_000) - 1, 1 - rng.random(10000) * 1e-6, rng.normal(0, 1e-9, 1000),
                        [-1.0, 1.0, 0.0, -0.0, 0.5, -0.5, 0.975, -0.975, 2.0 ** -26, 1e-310, 1.5, np.nan]])
    ref = np.empty_like(x)
    _asin_many(x, ref)
    got = asin_tensor(torch.from_numpy(x).to(device)).cpu().numpy()
    assert np.array_equal(bits(got), bits(ref))


@pytest.mark.parametrize("device,compile", list(_modes()))
@pytest.mark.parametrize("compression", [20, 250, 1000])
def test_fit_batch_matches_numba(device, compile, compression):
    _skip_compile(device, compile)
    names, x = short_matrix()
    batch = fit_tdigest_batch(torch.from_numpy(x).to(device), compression, compile=compile)
    assert batch.mean.device.type == device
    for name, row, state in zip(names, x, batch_states(batch)):
        assert_state_equal(state, tc.case_single(TDigest, compression, row), f"{name}|c{compression}")


@pytest.mark.parametrize("device,compile", list(_modes()))
@pytest.mark.parametrize("variant", ["refit_uneq7", "add_seq_eq4", "add_tree_eq8"])
def test_merge_batch_matches_numba(device, compile, variant):
    _skip_compile(device, compile)
    compression = 100
    names, x = short_matrix()
    if variant == "refit_uneq7":
        cuts = tc.unequal_split(np.arange(x.shape[1]), 7)
        parts = [x[:, c[0]:c[-1] + 1] if c.size else x[:, :0] for c in cuts]
    else:
        parts = np.array_split(x, 4 if variant == "add_seq_eq4" else 8, axis=1)

    def fit(p):
        return fit_tdigest_batch(torch.from_numpy(np.ascontiguousarray(p)).to(device), compression,
                                 compile=compile)

    def rebuild(batch):
        # round-trip through the per-channel state the scalers store, like attributes_to_td_objs
        return TDigestBatch.from_states(*batch.to_states(), compression, device=device)

    if variant == "refit_uneq7":
        acc = fit(parts[0])
        for p in parts[1:]:
            acc = merge_tdigest_batch(rebuild(acc), fit(p), compile=compile)
    elif variant == "add_seq_eq4":
        batches = [fit(p) for p in parts]
        acc = batches[0]
        for b in batches[1:]:
            acc = merge_tdigest_batch(rebuild(acc), rebuild(b), compile=compile)
    else:
        batches = [fit(p) for p in parts]
        while len(batches) > 1:
            batches = [merge_tdigest_batch(rebuild(batches[i]), rebuild(batches[i + 1]), compile=compile)
                       for i in range(0, len(batches), 2)]
        acc = batches[0]
    for i, (name, state) in enumerate(zip(names, batch_states(acc))):
        if variant == "refit_uneq7":
            ref = tc.case_refit(TDigest, compression, [p[i] for p in parts])
        elif variant == "add_seq_eq4":
            ref = tc.case_add_sequential(TDigest, compression, [p[i] for p in parts])
        else:
            ref = tc.case_add_tree(TDigest, compression, [p[i] for p in parts])
        assert_state_equal(state, ref, f"{name}|{variant}")


@pytest.mark.parametrize("device", DEVICES)
def test_fit_batch_matches_crick_fixture(device):
    """Full-length fixture data through the torch fit reproduces crick's stored output (torch == numba == crick)."""
    with np.load(FIXTURE) as f:
        reference = {k: f[k] for k in f.files}
    names = [name for name in tc.DATASETS if reference[f"{name}|data"].size == tc.N]
    x = np.stack([reference[f"{name}|data"] for name in names]).astype(np.float64)
    batch = fit_tdigest_batch(torch.from_numpy(x).to(device), 250)
    for name, state in zip(names, batch_states(batch)):
        key = tc.case_key(name, 250, "single")
        assert_state_equal(state, (reference[key + "|centroids"], *reference[key + "|scalars"]), key)


@pytest.mark.parametrize("device", DEVICES)
def test_static_rounds_fallback_is_exact(device):
    """A too-small static round bound flags channels on device; they are recomputed with numba, still exact."""
    names, x = short_matrix(3000)
    batch = fit_tdigest_batch(torch.from_numpy(x).to(device), 250, rounds=3)
    for name, row, state in zip(names, x, batch_states(batch)):
        assert_state_equal(state, tc.case_single(TDigest, 250, row), name)
    parts = np.array_split(x, 2, axis=1)
    a = fit_tdigest_batch(torch.from_numpy(np.ascontiguousarray(parts[0])).to(device), 250)
    b = fit_tdigest_batch(torch.from_numpy(np.ascontiguousarray(parts[1])).to(device), 250)
    merged = merge_tdigest_batch(a, b, rounds=2)
    for i, (name, state) in enumerate(zip(names, batch_states(merged))):
        # batches are always flushed, so the numba reference merges flushed (state round-tripped) digests
        assert_state_equal(state, tc.case_add_sequential(TDigest, 250, [parts[0][i], parts[1][i]]), name)


@pytest.mark.parametrize("device", DEVICES)
def test_fit_batch_edge_inputs(device):
    rng = np.random.default_rng(11)
    rows = [np.full(500, np.nan), np.concatenate([[np.inf], rng.normal(size=499)]), np.arange(500.0),
            np.where(rng.random(500) < 0.9, np.nan, rng.normal(size=500)), np.zeros(500), -np.zeros(500)]
    x = np.stack(rows)
    for dtype in [torch.float64, torch.float32]:
        xt = torch.from_numpy(x).to(device=device, dtype=dtype)
        batch = fit_tdigest_batch(xt, 50)
        for i, state in enumerate(batch_states(batch)):
            ref = tc.case_single(TDigest, 50, xt[i].cpu().numpy())
            assert_state_equal(state, ref, f"row{i}|{dtype}")


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("channels_last", [True, False])
def test_dquantile_scaler_tensor_torch_fit_matches_numba(device, channels_last):
    from bridgescaler.distributed_tensor import DQuantileScalerTensor
    rng = np.random.default_rng(8)
    shape = (20, 9, 11, 4) if channels_last else (20, 4, 9, 11)
    data = rng.gamma(1.5, size=shape)
    data[rng.random(shape) < 0.02] = np.nan
    names = ["a", "b", "c", "d"]

    def tensor(arr, cols=names):
        t = torch.from_numpy(np.ascontiguousarray(arr)).to(device)
        t.variable_names = list(cols)
        return t

    scalers = {}
    for backend in ["numba", "torch"]:
        s = DQuantileScalerTensor(compression=100, channels_last=channels_last, fit_backend=backend)
        s.fit(tensor(data[:10]))
        # refit with reordered columns exercises the channel permutation on merge
        perm = [2, 0, 3, 1]
        part = data[10:][..., perm] if channels_last else data[10:][:, perm]
        s.fit(tensor(part, [names[p] for p in perm]))
        scalers[backend] = s
    for i in range(len(names)):
        ref = (scalers["numba"].centroids_[i], scalers["numba"].size_[i], scalers["numba"].min_[i],
               scalers["numba"].max_[i])
        got = (scalers["torch"].centroids_[i], scalers["torch"].size_[i], scalers["torch"].min_[i],
               scalers["torch"].max_[i])
        assert_state_equal(got, ref, names[i])
    x = tensor(data)
    torch.testing.assert_close(scalers["torch"].transform(x).cpu(), scalers["numba"].transform(x).cpu(),
                               rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("flushes_per_call", [2, 3])
def test_flush_blocks_match_numba(flushes_per_call):
    """Several flushes per call (as compiled blocks run them), including the merge's min/max timing, stay exact."""
    names, x = short_matrix(4000)
    parts = np.array_split(x, 3, axis=1)
    batches = [fit_tdigest_batch(torch.from_numpy(np.ascontiguousarray(p)), 20, flushes_per_call=flushes_per_call)
               for p in parts]
    acc = batches[0]
    for b in batches[1:]:
        acc = merge_tdigest_batch(TDigestBatch.from_states(*acc.to_states(), 20),
                                  TDigestBatch.from_states(*b.to_states(), 20), flushes_per_call=flushes_per_call)
    for i, (name, state) in enumerate(zip(names, batch_states(acc))):
        assert_state_equal(state, tc.case_add_sequential(TDigest, 20, [p[i] for p in parts]), name)
