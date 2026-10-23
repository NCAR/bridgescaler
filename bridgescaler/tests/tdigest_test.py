import os

import numpy as np
import pytest

import tdigest_cases as tc
from bridgescaler.tdigest import TDigest
from bridgescaler.distributed import DQuantileScaler, tdigest_cdf, tdigest_quantile

FIXTURE = os.path.join(os.path.dirname(__file__), "data", "tdigest_reference.npz")


@pytest.fixture(scope="module")
def reference():
    with np.load(FIXTURE) as f:
        return {k: f[k] for k in f.files}


def bits(a):
    """View float64 values as integers so the comparison is exact (distinguishes -0.0/0.0, compares NaN)."""
    return np.ascontiguousarray(np.asarray(a, dtype=np.float64)).view(np.uint64)


def assert_state_equal(state, ref_centroids, ref_scalars, label):
    centroids, size, t_min, t_max = state
    assert centroids.shape == ref_centroids.shape, f"{label}: centroid count {centroids.shape} != {ref_centroids.shape}"
    n_diff = int(np.count_nonzero(bits(centroids) != bits(ref_centroids)))
    assert n_diff == 0, f"{label}: {n_diff} centroid values differ from reference"
    assert np.array_equal(bits([size, t_min, t_max]), bits(ref_scalars)), \
        f"{label}: size/min/max {[size, t_min, t_max]} != {ref_scalars}"


def crick_state(td):
    return tc.to_state(td)


@pytest.mark.parametrize("name", list(tc.DATASETS))
def test_port_matches_crick_fixture(reference, name):
    """The port reproduces crick's stored output bit-for-bit for every compression and fitting variant."""
    x = reference[f"{name}|data"]
    for c in tc.COMPRESSIONS:
        for variant in tc.VARIANTS:
            key = tc.case_key(name, c, variant)
            state = tc.run_case(TDigest, name, c, variant, x)
            assert_state_equal(state, reference[key + "|centroids"], reference[key + "|scalars"], key)


def test_port_matches_live_crick():
    """Fresh random data (not the fixture seeds) compared directly against crick when it is installed."""
    crick = pytest.importorskip("crick")
    rng = np.random.default_rng(20261022)
    compressions = [19, 20, 37.3, 100, 250, 600.5, 1000, 1500]
    n_checked = 0
    for trial in range(3):
        for name, gen in tc.DATASETS.items():
            x = gen(rng, int(rng.integers(1, 40000)))
            c = compressions[int(rng.integers(len(compressions)))]
            for variant, fn in tc.VARIANTS.items():
                ref = fn(crick.TDigest, c, x)
                state = fn(TDigest, c, x)
                assert_state_equal(state, ref[0], np.array(ref[1:], dtype=np.float64),
                                   f"trial{trial}|{name}|c{c}|{variant}")
                n_checked += 1
    # Weights, multidimensional / non-contiguous input (crick follows NpyIter KEEPORDER, which flips negative
    # strides) and repeated update calls on one digest.
    x = rng.gamma(2.0, size=(60, 70, 3))
    chunks = [x[..., 0], x[::2, ::-1, 1], x[::-1, ::-3, 2], np.asfortranarray(x[..., 2]), x[..., 0].T, x,
              x[5:9, 3, 1], x[..., 1].astype(np.float32)]
    weights = [1, 2.5, 1e-3, 1e-17, lambda c: rng.uniform(0.1, 3.0, size=c.shape),
               lambda c: rng.uniform(0.1, 3.0, size=c.shape[-1:])]
    for wi, w in enumerate(weights):
        a, b = crick.TDigest(250), TDigest(250)
        for chunk in chunks:
            wc = w(chunk) if callable(w) else w
            a.update(chunk, wc)
            b.update(chunk, wc)
        ref = tc.to_state(a)
        assert_state_equal(tc.to_state(b), ref[0], np.array(ref[1:]), f"weights[{wi}]")
    assert n_checked > 0


ACCURACY_DATASETS = ["gaussian", "gamma_0.5", "gamma_5", "bimodal", "trimodal", "lognormal", "uniform",
                     "exponential", "student_t3", "cauchy", "multiscale", "with_nonfinite", "float32", "sorted",
                     "reverse_sorted"]


@pytest.mark.parametrize("name", ACCURACY_DATASETS)
@pytest.mark.parametrize("variant", ["merge_eq2", "merge_eq7", "merge_uneq4", "refit_uneq7", "add_seq_eq4",
                                     "add_tree_eq8"])
def test_single_vs_combined(name, variant):
    """Combining digests fit to partitions is not bitwise equal to a single fit, but must be as accurate."""
    c = 250
    x = tc.make_data(name).astype(np.float64)
    xf = np.sort(x[np.isfinite(x)])
    single = tc.run_case(TDigest, name, c, "single", x)
    combined = tc.run_case(TDigest, name, c, variant, x)
    # total weight and extremes are exact regardless of how the data were partitioned
    assert single[1] == combined[1] == xf.size
    assert single[2] == combined[2] == xf[0]
    assert single[3] == combined[3] == xf[-1]

    # Rank error of quantile estimates against the empirical distribution. multiscale puts half its mass in a
    # spike 1e-10 wide, where tiny quantile errors become large rank jumps, so it gets a looser interior bound.
    mid_tol = 0.02 if name == "multiscale" else 0.01
    p_mid = np.linspace(0.01, 0.99, 99)
    p_tail = np.array([1e-3, 3e-3, 0.997, 0.999])

    def rank_err(state, p):
        q = tdigest_quantile(p, state[0][:, 0], state[0][:, 1], state[2], state[3])
        return np.abs(np.searchsorted(xf, q) / xf.size - p).max()

    for label, state in [("single", single), (variant, combined)]:
        assert rank_err(state, p_mid) <= mid_tol, f"{label} interior rank error too large"
        assert rank_err(state, p_tail) <= 1e-3, f"{label} tail rank error too large"

    # The CDFs of the two digests agree with each other at the data's own quantiles.
    probes = np.quantile(xf, np.linspace(0.005, 0.995, 199))
    cdf_single = tdigest_cdf(probes, single[0][:, 0], single[0][:, 1], single[2], single[3])
    cdf_combined = tdigest_cdf(probes, combined[0][:, 0], combined[0][:, 1], combined[2], combined[3])
    assert np.abs(cdf_single - cdf_combined).max() <= 2 * mid_tol


SCALER_COLUMNS = ["gaussian", "gamma_0.5", "bimodal", "trimodal", "cauchy", "integers", "with_nonfinite",
                  "float32", "signed_zeros"]


def _scaler_matrix(reference):
    return np.stack([reference[f"{n}|data"].astype(np.float64) for n in SCALER_COLUMNS], axis=-1)


def _assert_scaler_matches(scaler, reference, variant, c=250):
    for i, name in enumerate(SCALER_COLUMNS):
        key = tc.case_key(name, c, variant)
        state = (scaler.centroids_[i], scaler.size_[i], scaler.min_[i], scaler.max_[i])
        assert_state_equal(state, reference[key + "|centroids"], reference[key + "|scalars"],
                           f"{type(scaler).__name__}|{key}")


def test_dquantile_scaler_matches_crick_fixture(reference):
    x = _scaler_matrix(reference)
    single = DQuantileScaler(compression=250)
    single.fit(x)
    _assert_scaler_matches(single, reference, "single")

    refit = DQuantileScaler(compression=250)
    for part in np.array_split(x, 4):
        refit.fit(part)
    _assert_scaler_matches(refit, reference, "refit_eq4")

    parts = []
    for part in np.array_split(x, 4):
        parts.append(DQuantileScaler(compression=250))
        parts[-1].fit(part)
    added = parts[0]
    for p in parts[1:]:
        added = added + p
    _assert_scaler_matches(added, reference, "add_seq_eq4")


def test_dquantile_scaler_tensor_matches_crick_fixture(reference):
    torch = pytest.importorskip("torch")
    from bridgescaler.distributed_tensor import DQuantileScalerTensor
    x = torch.from_numpy(_scaler_matrix(reference))
    x.variable_names = list(SCALER_COLUMNS)
    single = DQuantileScalerTensor(compression=250)
    single.fit(x)
    _assert_scaler_matches(single, reference, "single")

    refit = DQuantileScalerTensor(compression=250)
    for part in torch.tensor_split(x, 4):
        part.variable_names = list(SCALER_COLUMNS)
        refit.fit(part)
    _assert_scaler_matches(refit, reference, "refit_eq4")

    parts = []
    for part in torch.tensor_split(x, 4):
        part.variable_names = list(SCALER_COLUMNS)
        parts.append(DQuantileScalerTensor(compression=250))
        parts[-1].fit(part)
    added = parts[0]
    for p in parts[1:]:
        added = added + p
    _assert_scaler_matches(added, reference, "add_seq_eq4")


def test_dquantile_scaler_channels_first_matches_direct_fit():
    rng = np.random.default_rng(5)
    x = np.stack([rng.normal(size=(40, 6, 7)), rng.gamma(0.7, size=(40, 6, 7))], axis=1)
    scaler = DQuantileScaler(compression=100, channels_last=False)
    scaler.fit(x)
    for i in range(x.shape[1]):
        state = tc.to_state(tc.fit(TDigest, 100, x[:, i]))
        assert_state_equal((scaler.centroids_[i], scaler.size_[i], scaler.min_[i], scaler.max_[i]),
                           state[0], np.array(state[1:]), f"channel {i}")


def test_tdigest_api_edge_cases():
    td = TDigest(100)
    assert np.isnan(td.min()) and np.isnan(td.max()) and td.size() == 0
    assert td.centroids().shape == (0,)
    td.update(np.array([], dtype=np.float64))
    td.update(np.array([np.nan, np.inf, -np.inf]))
    assert td.size() == 0
    td.update(np.arange(10, dtype=np.int32))
    assert td.size() == 10 and td.min() == 0 and td.max() == 9
    with pytest.raises(TypeError):
        td.update(np.array(["a", "b"]))
    with pytest.raises(ValueError):
        td.update(np.arange(3.0), w=-1)
    with pytest.raises(ValueError):
        TDigest(np.inf)
    assert TDigest(5).compression == 20 and TDigest(5000).compression == 1000
    clone = TDigest(100)
    clone.__setstate__(td.__getstate__())
    assert np.array_equal(clone.centroids(), td.centroids()) and clone.size() == td.size()


@pytest.mark.parametrize("channel_axis", [-1, 1])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_tdigest_arrays_match_per_channel(channel_axis, dtype):
    """The prange batch fit/merge equals one TDigest per channel, fit on ``np.take(x, i, axis).ravel()``."""
    from bridgescaler.tdigest import TDigestArrays
    rng = np.random.default_rng(4)
    x = rng.gamma(1.5, size=(30, 7, 12, 9)).astype(dtype)
    x[rng.random(x.shape) < 0.02] = np.nan
    x[(slice(None),) * (channel_axis % x.ndim) + (2,)] = np.nan  # an all-NaN channel
    n = x.shape[channel_axis]
    first, second = x[:13], x[13:]
    batch = TDigestArrays.fit(first, 100, channel_axis=channel_axis)
    batch.merge(TDigestArrays.fit(second, 100, channel_axis=channel_axis))
    cents, size, t_min, t_max = batch.to_states()
    for i in range(n):
        td = TDigest(100)
        td.update(np.take(first, i, channel_axis).ravel())
        td.centroids()
        other = TDigest(100)
        other.update(np.take(second, i, channel_axis).ravel())
        td.merge(other)
        assert_state_equal((cents[i], size[i], t_min[i], t_max[i]), tc.to_state(td)[0],
                           np.array(tc.to_state(td)[1:]), f"channel {i}")


def test_tdigest_arrays_merge_targets():
    from bridgescaler.tdigest import TDigestArrays
    rng = np.random.default_rng(6)
    x = rng.normal(size=(5000, 5))
    y = rng.normal(2, 1, size=(4000, 3))
    batch = TDigestArrays.fit(x, 250)
    batch.merge(TDigestArrays.fit(y, 250), targets=[4, 0, 2])
    cents = batch.to_states()[0]
    for i in range(5):
        td = TDigest(250)
        td.update(x[:, i].copy())
        td.centroids()
        if i in (4, 0, 2):
            other = TDigest(250)
            other.update(y[:, [4, 0, 2].index(i)].copy())
            td.merge(other)
        assert np.array_equal(bits(cents[i]), bits(tc.to_state(td)[0])), f"channel {i}"
    with pytest.raises(ValueError):
        batch.merge(TDigestArrays.fit(y, 250), targets=[0, 0, 1])


def test_dquantile_scaler_add_subset_columns():
    import pandas as pd
    rng = np.random.default_rng(9)
    a = pd.DataFrame(rng.normal(size=(3000, 4)), columns=list("abcd"))
    b = pd.DataFrame(rng.gamma(2.0, size=(2000, 2)), columns=["d", "b"])
    sa, sb = DQuantileScaler(compression=100), DQuantileScaler(compression=100)
    sa.fit(a)
    sb.fit(b)
    combined = sa + sb
    for col in "abcd":
        i = "abcd".index(col)
        td = TDigest(100)
        td.update(a[col].values)
        td.centroids()
        if col in ("d", "b"):
            other = TDigest(100)
            other.update(b[col].values)
            td.merge(other)
        ref = tc.to_state(td)
        assert_state_equal((combined.centroids_[i], combined.size_[i], combined.min_[i], combined.max_[i]),
                           ref[0], np.array(ref[1:]), col)
