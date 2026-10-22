from bridgescaler import save_scaler, load_scaler, save_scaler_dict, load_scaler_dict
from bridgescaler.distributed_tensor import (DStandardScalerTensor, DMinMaxScalerTensor, DQuantileScalerTensor,
                                            DSpatialStandardScalerTensor)
import time
import warnings
import copy
import numpy as np
import pytest
import torch
import os
import functools


@functools.lru_cache(maxsize=1)
def _inductor_cpu_available():
    """Probe whether torch.compile's Inductor CPU backend can actually build a kernel.

    On some HPC module environments the ``g++`` on PATH is a wrapper (e.g. NCAR's
    ``ncarcompilers``) that injects link flags, which breaks Inductor's header-only
    precompile step and raises a CppCompileError. That is a toolchain issue, not a
    bridgescaler bug, so tests that require compilation skip rather than fail.
    Set ``CXX`` to a plain g++ to make this probe (and the compile path) succeed.
    """
    try:
        fn = torch.compile(lambda t: t * 2 + 1, fullgraph=True)
        fn(torch.randn(4))
    except Exception:
        return False
    return True


def make_test_data():
    np.random.seed(34325)
    test_data = dict()
    test_data["means"] = np.array([0, 5.3, -2.421, 21456.3, 1.e-5])
    test_data["sds"] = np.array([5, 352.2, 1e-4, 20000.3, 5.3e-2])
    test_data["n_examples"] = np.array([1000, 500, 88])
    test_data["numpy_2d"] = []
    test_data["numpy_4d"] = []
    tile_width = 5
    for n in range(test_data["n_examples"].size):
        data2d = np.zeros((test_data["n_examples"][n], test_data["means"].size))
        data4d = np.zeros((test_data["n_examples"][n], tile_width, tile_width, test_data["means"].size))
        for i in range(test_data["means"].size):
            data2d[:, i] = np.random.normal(loc=test_data["means"][i],
                                            scale=test_data["sds"][i],
                                            size=test_data["n_examples"][n])
            data4d[..., i] = np.random.normal(loc=test_data["means"][i],
                                              scale=test_data["sds"][i],
                                              size=(test_data["n_examples"][n], tile_width, tile_width))
        test_data["numpy_2d"].append(data2d)
        test_data["numpy_4d"].append(data4d)

    return test_data


# Create test datasets for use in all unit tests.
test_data = make_test_data()

def test_dstandard_tensor_scaler():
    numpy_2d_1 = torch.from_numpy(test_data["numpy_2d"][0])
    numpy_2d_2 = torch.from_numpy(test_data["numpy_2d"][1])
    numpy_2d_3 = torch.from_numpy(test_data["numpy_2d"][2])
    all_ds_2d = torch.vstack([numpy_2d_1, numpy_2d_2, numpy_2d_3])
    numpy_4d_1 = torch.from_numpy(test_data["numpy_4d"][0])
    numpy_4d_2 = torch.from_numpy(test_data["numpy_4d"][1])
    numpy_4d_3 = torch.from_numpy(test_data["numpy_4d"][2])
    all_ds_4d = torch.vstack([numpy_4d_1, numpy_4d_2, numpy_4d_3])
    dsses_2d = []
    dsses_4d = []
    for n in range(test_data["n_examples"].size):
        dsses_2d.append(DStandardScalerTensor())
        dsses_2d[-1].fit(torch.from_numpy(test_data["numpy_2d"][n]))
        dsses_4d.append(DStandardScalerTensor(channels_last=True))
        dsses_4d[-1].fit(torch.from_numpy(test_data["numpy_4d"][n]))
        save_scaler(dsses_2d[-1], "scaler.json")
        new_scaler = load_scaler("scaler.json")
        os.remove("scaler.json")
    dss_total_2d = dsses_2d[0] + dsses_2d[1] + dsses_2d[2]
    dss_total_4d = dsses_4d[0] + dsses_4d[1] + dsses_4d[2]
    mean_2d, var_2d = dss_total_2d.get_scales()
    mean_4d, var_4d = dss_total_4d.get_scales()
    all_2d_var = all_ds_2d.var(axis=0, unbiased=False)
    all_4d_var = torch.tensor([all_ds_4d[..., i].var(unbiased=False) for i in range(all_ds_4d.shape[-1])])
    all_4d_mean = torch.tensor([all_ds_4d[..., i].mean() for i in range(all_ds_4d.shape[-1])])
    assert mean_2d.shape[0] == test_data["means"].shape[0] and var_2d.shape[0] == test_data["sds"].shape[0], "Stat shape mismatch"
    assert mean_4d.shape[0] == test_data["means"].shape[0] and var_4d.shape[0] == test_data["sds"].shape[0], "Stat shape mismatch"
    assert torch.max(torch.abs(mean_2d - all_ds_2d.mean(axis=0))) < 1e-5, "significant difference in means"
    assert torch.max(torch.abs(var_2d - all_2d_var) / all_2d_var) < 1e-5, "significant difference in variances"
    assert torch.max(torch.abs(mean_4d - all_4d_mean) / all_4d_mean) < 1e-5, "significant difference in means"
    assert torch.max(torch.abs(var_4d - all_4d_var) / all_4d_var) < 1e-5, "significant difference in variances"


def test_dstandard_scaler_nan_aware():
    # A channel with real physical NaNs (e.g. SST over land) must not poison its own mean/var, and
    # must match what you'd get by manually stripping the NaNs out and fitting on what's left.
    torch.manual_seed(0)
    n = 5000
    is_valid = torch.rand(n) > 0.3  # ~30% NaN, like land fraction in an SST field
    sst = torch.full((n,), float("nan"), dtype=torch.float64)
    sst[is_valid] = torch.randn(int(is_valid.sum()), dtype=torch.float64) * 4 + 290
    clean = torch.randn(n, dtype=torch.float64)
    x = torch.stack([clean, sst], dim=1)
    x.variable_names = ["clean", "sst"]

    scaler_nan = DStandardScalerTensor(channels_last=True)
    scaler_nan.fit(x)
    mean_nan, var_nan = scaler_nan.get_scales()
    sst_idx = scaler_nan.x_columns_.index("sst")
    clean_idx = scaler_nan.x_columns_.index("clean")

    # Same "sst" data with the NaN entries removed beforehand -- the ground truth the internal
    # NaN-masking must match.
    sst_stripped = sst[is_valid].reshape(-1, 1)
    sst_stripped.variable_names = ["sst"]
    scaler_stripped = DStandardScalerTensor(channels_last=True)
    scaler_stripped.fit(sst_stripped)
    mean_stripped, var_stripped = scaler_stripped.get_scales()

    assert torch.isclose(mean_nan[sst_idx], mean_stripped[0], atol=1e-8), \
        "mean differs between NaN-laced fit and manually NaN-stripped fit"
    assert torch.isclose(var_nan[sst_idx], var_stripped[0], atol=1e-8), \
        "variance differs between NaN-laced fit and manually NaN-stripped fit"
    assert scaler_nan.n_[sst_idx] == int(is_valid.sum()), "count should equal number of valid (non-NaN) entries"

    # The NaN-free channel must be completely unaffected (regression check against plain mean/var).
    assert torch.isclose(mean_nan[clean_idx], clean.mean(), atol=1e-8)
    assert torch.isclose(var_nan[clean_idx], clean.var(correction=0), atol=1e-8)
    assert scaler_nan.n_[clean_idx] == n

    # An all-NaN channel yields NaN stats and a zero count, without raising.
    x_allnan = torch.stack([clean, torch.full((n,), float("nan"), dtype=torch.float64)], dim=1)
    x_allnan.variable_names = ["clean", "all_nan"]
    scaler_edge = DStandardScalerTensor(channels_last=True)
    scaler_edge.fit(x_allnan)
    mean_edge, var_edge = scaler_edge.get_scales()
    edge_idx = scaler_edge.x_columns_.index("all_nan")
    assert torch.isnan(mean_edge[edge_idx]) and torch.isnan(var_edge[edge_idx])
    assert scaler_edge.n_[edge_idx] == 0

    # transform (unmodified) still just passes NaN through elementwise at that one position.
    row = torch.tensor([[0.5, float("nan")], [0.5, 300.0]], dtype=torch.float64)
    row.variable_names = ["clean", "sst"]
    out = scaler_nan.transform(row)
    assert torch.isnan(out[0, 1]) and not torch.isnan(out[0, 0]), "NaN should not spread to other channels"
    assert not torch.isnan(out[1, 1]), "clean value in the same channel should transform normally"


def test_dstandard_scaler_merge_masked_channel():
    # A channel can be entirely absent from one batch and present in the next (seasonal sea ice in a
    # per-gridpoint fit, say). Merging must pick up whichever side has data instead of letting the
    # empty side's NaN mean/var through: n=0 does not mask it out on its own, because 0 * nan is nan.
    torch.manual_seed(0)
    n = 400
    names = ["always", "seasonal", "never"]

    def make_batch(seasonal_valid):
        always = torch.randn(n, dtype=torch.float64) * 3 + 10
        seasonal = (
            torch.randn(n, dtype=torch.float64) * 2 + 270
            if seasonal_valid
            else torch.full((n,), float("nan"), dtype=torch.float64)
        )
        never = torch.full((n,), float("nan"), dtype=torch.float64)
        x = torch.stack([always, seasonal, never], dim=1)
        x.variable_names = list(names)
        return x

    masked = make_batch(seasonal_valid=False)  # "seasonal" fully NaN here
    valid = make_batch(seasonal_valid=True)

    # Ground truth: a single fit over both batches' samples at once.
    both = torch.cat([masked, valid], dim=0)
    both.variable_names = list(names)
    oracle = DStandardScalerTensor(channels_last=True)
    oracle.fit(both)
    o_mean, o_var = oracle.get_scales()

    def check(scaler, label):
        mean, var = scaler.get_scales()
        for name in ("always", "seasonal"):
            i = scaler.x_columns_.index(name)
            assert torch.isfinite(mean[i]), f"{label}: '{name}' mean went non-finite"
            assert torch.isfinite(var[i]), f"{label}: '{name}' var went non-finite"
            assert torch.isclose(mean[i], o_mean[i], atol=1e-8), \
                f"{label}: '{name}' mean does not match a single fit over both batches"
            assert torch.isclose(var[i], o_var[i], atol=1e-8), \
                f"{label}: '{name}' var does not match a single fit over both batches"
            assert scaler.n_[i] == oracle.n_[i], f"{label}: '{name}' count does not match"
        # Observed nowhere, so there is no mean to report.
        j = scaler.x_columns_.index("never")
        assert torch.isnan(mean[j]) and torch.isnan(var[j]), \
            f"{label}: a channel empty in every batch should stay NaN"
        assert scaler.n_[j] == 0, f"{label}: 'never' count should be zero"

    # __add__ (the map-reduce path: per-rank or per-file fits summed together).
    sa = DStandardScalerTensor(channels_last=True)
    sa.fit(masked)
    sb = DStandardScalerTensor(channels_last=True)
    sb.fit(valid)
    check(sa + sb, "masked + valid")
    check(sb + sa, "valid + masked")

    # Repeated fit() on one scaler (the running-average path), in both orders.
    for first, second, label in ((masked, valid, "fit(masked), fit(valid)"),
                                 (valid, masked, "fit(valid), fit(masked)")):
        running = DStandardScalerTensor(channels_last=True)
        running.fit(first)
        running.fit(second)
        check(running, label)


def test_dstandard_scaler_merge_unchanged_without_masking():
    # Regression guard for the merge refactor: with data on both sides the combined moments must be
    # exactly the original expression, bit for bit, not merely close to it.
    torch.manual_seed(1)
    x_a = torch.randn(300, 4, dtype=torch.float64) * 5 + 2
    x_b = torch.randn(150, 4, dtype=torch.float64) * 5 + 2

    sa = DStandardScalerTensor(channels_last=True)
    sa.fit(x_a)
    sb = DStandardScalerTensor(channels_last=True)
    sb.fit(x_b)
    n_a, mean_a, var_a = sa.n_, sa.mean_x_, sa.var_x_
    n_b, mean_b, var_b = sb.n_, sb.mean_x_, sb.var_x_

    expected_mean = (n_a * mean_a + n_b * mean_b) / (n_a + n_b)
    expected_var = ((n_a * var_a + n_b * var_b) / (n_a + n_b)
                    + (n_a * n_b * (mean_a - mean_b) ** 2) / ((n_a + n_b) ** 2))

    merged = sa + sb
    assert torch.equal(merged.mean_x_, expected_mean), "merged mean changed for unmasked data"
    assert torch.equal(merged.var_x_, expected_var), "merged var changed for unmasked data"
    assert torch.equal(merged.n_, n_a + n_b)


def test_dminmax_scaler_nan_aware():
    torch.manual_seed(1)
    n = 5000
    is_valid = torch.rand(n) > 0.3
    sst = torch.full((n,), float("nan"), dtype=torch.float64)
    sst[is_valid] = torch.randn(int(is_valid.sum()), dtype=torch.float64) * 4 + 290
    clean = torch.randn(n, dtype=torch.float64)
    x = torch.stack([clean, sst], dim=1)
    x.variable_names = ["clean", "sst"]

    scaler_nan = DMinMaxScalerTensor(channels_last=True)
    scaler_nan.fit(x)
    min_nan, max_nan = scaler_nan.get_scales()
    sst_idx = scaler_nan.x_columns_.index("sst")
    clean_idx = scaler_nan.x_columns_.index("clean")

    sst_stripped = sst[is_valid].reshape(-1, 1)
    sst_stripped.variable_names = ["sst"]
    scaler_stripped = DMinMaxScalerTensor(channels_last=True)
    scaler_stripped.fit(sst_stripped)
    min_stripped, max_stripped = scaler_stripped.get_scales()

    assert torch.isclose(min_nan[sst_idx], min_stripped[0], atol=1e-8), \
        "min differs between NaN-laced fit and manually NaN-stripped fit"
    assert torch.isclose(max_nan[sst_idx], max_stripped[0], atol=1e-8), \
        "max differs between NaN-laced fit and manually NaN-stripped fit"

    # The NaN-free channel must be completely unaffected (regression check).
    assert torch.isclose(min_nan[clean_idx], clean.min())
    assert torch.isclose(max_nan[clean_idx], clean.max())

    # An all-NaN channel yields +inf/-inf (min > max, marking the channel as never having a valid
    # value), without raising.
    x_allnan = torch.stack([clean, torch.full((n,), float("nan"), dtype=torch.float64)], dim=1)
    x_allnan.variable_names = ["clean", "all_nan"]
    scaler_edge = DMinMaxScalerTensor(channels_last=True)
    scaler_edge.fit(x_allnan)
    min_edge, max_edge = scaler_edge.get_scales()
    edge_idx = scaler_edge.x_columns_.index("all_nan")
    assert torch.isinf(min_edge[edge_idx]) and torch.isinf(max_edge[edge_idx])

    row = torch.tensor([[0.5, float("nan")], [0.5, 300.0]], dtype=torch.float64)
    row.variable_names = ["clean", "sst"]
    out = scaler_nan.transform(row)
    assert torch.isnan(out[0, 1]) and not torch.isnan(out[0, 0]), "NaN should not spread to other channels"
    assert not torch.isnan(out[1, 1]), "clean value in the same channel should transform normally"


def test_dminmax_tensor_scaler():
    numpy_2d_1 = torch.from_numpy(test_data["numpy_2d"][0])
    numpy_2d_2 = torch.from_numpy(test_data["numpy_2d"][1])
    numpy_2d_3 = torch.from_numpy(test_data["numpy_2d"][2])
    all_ds_2d = torch.vstack([numpy_2d_1, numpy_2d_2, numpy_2d_3])
    numpy_4d_1 = torch.from_numpy(test_data["numpy_4d"][0])
    numpy_4d_2 = torch.from_numpy(test_data["numpy_4d"][1])
    numpy_4d_3 = torch.from_numpy(test_data["numpy_4d"][2])
    all_ds_4d = torch.vstack([numpy_4d_1, numpy_4d_2, numpy_4d_3])
    dsses_2d = []
    dsses_4d = []
    for n in range(test_data["n_examples"].size):
        dsses_2d.append(DMinMaxScalerTensor())
        dsses_2d[-1].fit(torch.from_numpy(test_data["numpy_2d"][n]))
        dsses_4d.append(DMinMaxScalerTensor())
        dsses_4d[-1].fit(torch.from_numpy(test_data["numpy_4d"][n]))
        save_scaler(dsses_2d[-1], "scaler.json")
        new_scaler = load_scaler("scaler.json")
        os.remove("scaler.json")
    dss_total_2d = dsses_2d[0] + dsses_2d[1] + dsses_2d[2]
    dss_total_4d = dsses_4d[0] + dsses_4d[1] + dsses_4d[2]
    min_2d, max_2d = dss_total_2d.get_scales()
    min_4d, max_4d = dss_total_4d.get_scales()
    assert torch.max(torch.abs(min_2d - all_ds_2d.min(axis=0).values)) < 1e-8, "significant difference in minimum"
    assert torch.max(torch.abs(max_2d - all_ds_2d.max(axis=0).values)) < 1e-8, "significant difference in maximum"
    assert torch.max(torch.abs(min_4d - all_ds_4d.amin(dim=(0, 1, 2)))) < 1e-8, "significant difference in minimum"
    assert torch.max(torch.abs(max_4d - all_ds_4d.amax(dim=(0, 1, 2)))) < 1e-8, "significant difference in maximum"


def test_dquantile_tensor_scaler():
    dsses_2d = []
    dsses_4d = []
    for n in range(test_data["n_examples"].size):
        dsses_2d.append(DQuantileScalerTensor())
        dsses_2d[-1].fit(torch.from_numpy(test_data["numpy_2d"][n]))
        dsses_4d.append(DQuantileScalerTensor())
        dsses_4d[-1].fit(torch.from_numpy(test_data["numpy_4d"][n]))
        ds_2d_transformed = dsses_2d[-1].transform(torch.from_numpy(test_data["numpy_2d"][n]))
        ds_4d_transformed = dsses_4d[-1].transform(torch.from_numpy(test_data["numpy_4d"][n]))
        ds_2d_it = dsses_2d[-1].inverse_transform(ds_2d_transformed)
        ds_4d_it = dsses_4d[-1].inverse_transform(ds_4d_transformed)
        assert ds_2d_transformed.max() <= 1, "Quantile transform > 1"
        assert ds_4d_transformed.max() <= 1, "Quantile transform > 1"
        save_scaler(dsses_2d[-1], "scaler.json")
        new_scaler = load_scaler("scaler.json")
        os.remove("scaler.json")
        assert torch.argmax(torch.abs(new_scaler.min_ - dsses_2d[-1].min_)) == 0, \
                    "Differences in scaler centroid values after loading"
        assert torch.all(~torch.isnan(ds_2d_transformed)), "nans in transform"
        assert torch.all(~torch.isnan(ds_2d_it)), "nans in inverse transform"
        assert ds_2d_transformed.shape == test_data["numpy_2d"][n].shape, "shape does not match"
        assert ds_2d_it.shape == test_data["numpy_2d"][n].shape, "shape does not match"
        assert torch.all(~torch.isnan(ds_4d_transformed)), "nans in transform"
        assert torch.all(~torch.isnan(ds_4d_it)), "nans in inverse transform"
        assert ds_4d_transformed.shape == test_data["numpy_4d"][n].shape, "shape does not match"
        assert ds_4d_it.shape == test_data["numpy_4d"][n].shape, "shape does not match"

    combined_scaler = dsses_2d[0] + dsses_2d[1] + dsses_2d[2]
    assert combined_scaler.size_[0] == test_data["n_examples"].sum(), \
        "Summing did not work properly."


def test_dquantile_tensor_vmap_edge_cases():
    # 5D (batch, channel, time, y, x) channels_first with per-variable scaling so centroid counts differ
    torch.manual_seed(11)
    x = torch.randn(4, 40, 6, 16, 16, dtype=torch.float64)
    for c in range(x.shape[1]):
        x[:, c] = x[:, c] * (c + 1) + c
    for dist in ["uniform", "normal", "logistic"]:
        sc = DQuantileScalerTensor(channels_last=False, distribution=dist)
        t = sc.fit_transform(x)
        it = sc.inverse_transform(t)
        # stacked centroid tensors are built with +inf mean padding and 0 weight padding
        assert sc.centroids_count.shape[0] == x.shape[1], "one centroid count per variable"
        assert torch.isinf(sc.centroids_mean_stacked).any(), "ragged centroids should be inf-padded"
        assert torch.all(~torch.isnan(t)) and torch.all(~torch.isnan(it)), f"nans for {dist}"
        assert t.shape == x.shape and it.shape == x.shape, "shape preserved"

    # single-centroid (constant / single-example) variables exercise the n_real <= 1 branch
    x1 = torch.tensor([[2.5, -1.0]], dtype=torch.float64)
    sc = DQuantileScalerTensor(distribution="uniform")
    t = sc.fit_transform(x1)
    it = sc.inverse_transform(t)
    assert sc.centroids_count.tolist() == [1, 1], "single-example fit yields one centroid per variable"
    assert torch.all(~torch.isnan(t)) and torch.all(~torch.isnan(it)), "nans in single-centroid path"
    assert torch.allclose(it, x1, atol=1e-6), "single-centroid inverse should recover the value"


def test_dquantile_tensor_refit_rebuilds_stacked_centroids():
    # Repeated fit on different subsets must rebuild the padded/stacked centroid tensors used by vmap,
    # otherwise transform would keep using stale centroids from the first subset only.
    torch.manual_seed(3)
    scale = torch.tensor([1., 5., 20., 100.], dtype=torch.float64)
    a = torch.randn(300, 4, dtype=torch.float64) * scale
    b = torch.randn(500, 4, dtype=torch.float64) * scale + 3.0
    a.variable_names = b.variable_names = ["v1", "v2", "v3", "v4"]

    sc = DQuantileScalerTensor(distribution="normal")
    sc.fit(a)
    stacked_1 = sc.centroids_mean_stacked.clone()
    counts_1 = sc.centroids_count.clone()

    sc.fit(b)  # incremental update branch
    # the stacked tensors must reflect the merged digest, not the first subset
    assert sc.centroids_count.numel() == 4
    assert (not torch.equal(counts_1, sc.centroids_count)
            or stacked_1.shape != sc.centroids_mean_stacked.shape
            or not torch.equal(stacked_1, sc.centroids_mean_stacked)), \
        "stacked centroids were not rebuilt after re-fitting on a new subset"
    # the stacked tensors must stay consistent with the ragged per-variable lists
    for o in range(4):
        k = sc.centroids_count[o].item()
        assert torch.allclose(sc.centroids_mean_stacked[o, :k], sc.centroids_mean_tensor[o]), \
            "stacked means diverged from the per-variable centroid list after re-fit"


def test_tensor_scaler_with_attribute():
    # create synthetic data (channel_dim=1)
    x_org = torch.randn(20, 5, 4, 8) * 2.2 # without attributes

    # add attributes
    x_attr = x_org.clone()
    x_attr.variable_names = ["ch1", "ch2", "ch3", "ch4", "ch5"]

    # reorder channels by swapping the last two channels
    x_swp = x_org[:, [0, 1, 2, 4, 3], :, :].clone()
    x_swp.variable_names = ["ch1", "ch2", "ch3", "ch5", "ch4"]

    # duplications in attributes
    x_dup = x_attr.clone()
    x_dup.variable_names = ["ch1", "ch2", "ch3", "ch4", "ch4"]

    # data (5 variables) and attribute (4 variables) mismatch
    x_mis = x_attr.clone()
    x_mis.variable_names = ["ch1", "ch2", "ch3", "ch4"]

    # data with different attributes
    x_dattr = x_attr.clone()
    x_dattr.variable_names = ["ch1", "ch2", "ch3", "ch4", "ch6"]
    
    ## data on GPU
    #x_gpu = x_attr.clone()
    #x_gpu = x_gpu.to("cuda:0")
    #x_gpu.variable_names = ["ch1", "ch2", "ch3", "ch4", "ch5"]

    # collect scalers to test
    scaler_objs = [DStandardScalerTensor, DSpatialStandardScalerTensor, DMinMaxScalerTensor, DQuantileScalerTensor]
    for scaler_class in scaler_objs:
        print("test on: ",scaler_class.__name__)

        # focus on "fitting"
        # test 1: verify that fitting data missing required attributes triggers a UserWarning.
        with pytest.warns(UserWarning,
                        match=r"^Input data lacks variable names. When performing fit or transform, "
                                r"data/scaler consistency check is limited to variable counts; order "
                                r"cannot be validated. Ensure variable alignment to prevent incorrect results.$"):
            scaler = scaler_class(channels_last=False)
            scaler.fit(x_org)

    # test 2: ensure that providing data with duplicate attributes raises a ValueError.
        with pytest.raises(ValueError,
                           match=r"^Duplicates found in variable_names! 4 unique vs 5 total.$"):
            scaler = scaler_class(channels_last=False)
            scaler.fit(x_dup)

        # test 3: ensure that providing data with mismatch attributes raises a ValueError.
        with pytest.raises(ValueError,
                           match=r"^Input data channel dimension mismatch: data has 5 channels \(dim=1\), but 4 were found in attribute variable_names.$"):
            scaler = scaler_class(channels_last=False)
            scaler.fit(x_mis)

        # test 4: validate that re-fitting is invariant to the permutation of input channel data and attributes.
        scaler = scaler_class(channels_last=False)
        scaler.fit(x_attr)
        scaler_swp = copy.deepcopy(scaler)
        scaler_swp.fit(x_swp)
        if scaler_class.__name__ == "DQuantileScalerTensor":
            pass  # re-fitting (even with the same data) could change the centroids since the population changes
        else:
            assert torch.all(scaler.get_scales()[0] == scaler_swp.get_scales()[0]), "scaler means are not identical"
            assert torch.all(scaler.get_scales()[1] == scaler_swp.get_scales()[1]), "scaler variances are not identical"

        # test 5: validate that re-fitting requires same attributes component
        with pytest.raises(AssertionError,
                           match=r"^Some input variables not in scaler x_columns. "
                                 r"Scaler: \['ch1', 'ch2', 'ch3', 'ch4', 'ch5'\], input variables: \['ch1', 'ch2', 'ch3', 'ch4', 'ch6'\]$"):
            scaler = scaler_class(channels_last=False)
            scaler.fit(x_attr)
            scaler.fit(x_dattr)

        # test 6: validate adding two scalers requires same variables
        with pytest.raises(AssertionError,
                           match=r"^Scaler variables do not match.$"):
            scaler1 = scaler_class(channels_last=False)
            scaler2 = scaler_class(channels_last=False)
            scaler1.fit(x_attr)
            scaler2.fit(x_dattr)
            scaler1 + scaler2

        # focus on "transform"
        # test 7: validate that transform is invariant to the permutation of input channel data and attributes.
        scaler = scaler_class(channels_last=False)
        assert torch.all(scaler.fit_transform(x_attr) == scaler.transform(x_swp)[
            :, [0, 1, 2, 4, 3], :, :]), "transformed data are not identical"

        # test 8: verify 'variable_names' attribute persists post-transform
        scaler = scaler_class(channels_last=False)
        out = scaler.fit_transform(x_attr)
        assert getattr(out, 'variable_names', None) is not None, "metadata 'variable_names' lost during transform"
        assert getattr(scaler.inverse_transform(out), 'variable_names',
                       None) is not None, "metadata 'variable_names' lost during inverse transform"

        # test9: verify consistency of the transformation with selective channels
        scaler = scaler_class(channels_last=False)
        x_sel = x_attr[:, [0, 1, 2, 4], :, :]
        x_sel.variable_names = ["ch1", "ch2", "ch3", "ch5"]

        assert torch.all(scaler.fit_transform(x_attr)[:, [0, 1, 2, 4], :, :] == scaler.transform(
            x_sel)), "transformation mismatch between full tensor slice and explicit channel subset."

        ## test10: verify tensor device
        #scaler = scaler_class(channels_last=False)
        #scaler.fit(x_gpu)
        #assert scaler.get_scales()[0].is_cuda, "device should be GPU"
        #assert scaler.transform(x_gpu).is_cuda, "device should be GPU"
        #assert scaler.transform(x_attr).is_cuda == False, "device should be CPU"


def test_dquantile_tensor_fast_transform():
    # fast_transform replaces the exact TDigest CDF/quantile eval with a per-variable monotone-cubic (PCHIP)
    # approximation in BOTH directions. Knots are lazy (built on first transform, invalidated on refit) and must
    # survive save/load, compose with channels_first + column subsetting, and match the exact path in the bulk.
    rng = np.random.default_rng(0)
    for dist in ["uniform", "normal", "logistic"]:
        data = np.stack([rng.lognormal(0, 1, 6000), rng.normal(3, 2, 6000),
                         rng.uniform(-1, 1, 6000), rng.exponential(2.0, 6000)], axis=1)
        x = torch.from_numpy(data.copy())
        exact = DQuantileScalerTensor(distribution=dist)
        fast = DQuantileScalerTensor(distribution=dist, fast_transform=True, n_knots=256)
        exact.fit(x)
        fast.fit(x)
        assert fast.knots_x_ is None, "knots should be lazy (unbuilt) right after fit"
        ze = exact.transform(x)
        zf = fast.transform(x)
        assert fast.knots_x_ is not None, "knots should be built on first transform"
        assert torch.all(~torch.isnan(zf)) and zf.shape == x.shape, f"bad fast transform for {dist}"
        # bulk (99th percentile) error vs the exact path is small; the extreme tail worst-case is larger by design
        assert torch.quantile(torch.abs(zf - ze), 0.99) < 0.05, f"fast_transform bulk error too high for {dist}"
        # fast inverse vs exact inverse (both applied to the exact-transform output), in data space
        xe = exact.inverse_transform(ze)
        xf = fast.inverse_transform(ze)
        assert torch.all(~torch.isnan(xf)) and xf.shape == x.shape, f"bad fast inverse for {dist}"
        tol = 0.05 * (xe.max() - xe.min())
        assert torch.quantile(torch.abs(xf - xe), 0.99) < tol, f"fast inverse bulk error too high for {dist}"

    # refit invalidates the cached knots so a stale curve is never reused
    fast.fit(x)
    assert fast.knots_x_ is None, "refit must invalidate fast_transform knots"

    # channels_first + column subsetting/reordering compose with the fast path
    x4 = torch.randn(400, 4, 4, 4, dtype=torch.float64)
    for c in range(4):
        x4[:, c] = x4[:, c] * (c + 1) + c
    x4.variable_names = ["a", "b", "c", "d"]
    fast_cf = DQuantileScalerTensor(distribution="normal", channels_last=False, fast_transform=True)
    full = fast_cf.fit_transform(x4)
    x_sel = x4[:, [0, 2, 1], :, :].clone()
    x_sel.variable_names = ["a", "c", "b"]
    assert torch.allclose(fast_cf.transform(x_sel), full[:, [0, 2, 1]], atol=1e-9), \
        "fast_transform must respect column subsetting/reordering"

    # save/load round-trip: lazy (knots None) and after knots are built
    data = torch.from_numpy(rng.lognormal(0, 1, (4000, 3)))
    s = DQuantileScalerTensor(distribution="normal", fast_transform=True)
    s.fit(data)
    save_scaler(s, "fast_scaler.json")
    s2 = load_scaler("fast_scaler.json")
    os.remove("fast_scaler.json")
    assert s2.fast_transform is True and int(s2.n_knots) == 256, "fast_transform config lost on load"
    assert s2.knots_x_ is None, "unbuilt knots should load back as None"
    assert torch.allclose(s2.transform(data), s.transform(data), atol=1e-9), "fast transform changed after load"
    save_scaler(s, "fast_scaler.json")  # now knots are built
    s3 = load_scaler("fast_scaler.json")
    os.remove("fast_scaler.json")
    assert torch.is_tensor(s3.knots_x_), "built knots should round-trip as a tensor"
    assert torch.allclose(s3.transform(data), s.transform(data), atol=1e-9), "fast transform changed after load"


def test_dquantile_tensor_compile_matches_eager():
    # compile=True uses torch.compile(fullgraph=True) on the vmapped kernel. fullgraph=True raises on any graph
    # break, so if this runs at all there is no graph break; we also assert it matches the eager path exactly.
    if not _inductor_cpu_available():
        pytest.skip("torch.compile Inductor CPU backend cannot build kernels in this environment "
                    "(likely a compiler-toolchain issue; set CXX to a plain g++)")
    torch.manual_seed(7)
    x = torch.randn(6, 8, 4, 5, dtype=torch.float64)
    for c in range(x.shape[1]):
        x[:, c] = x[:, c] * (c + 1) + c
    x.variable_names = [f"v{i}" for i in range(x.shape[1])]
    for dist in ["uniform", "normal", "logistic"]:
        eager = DQuantileScalerTensor(channels_last=False, distribution=dist)
        comp = DQuantileScalerTensor(channels_last=False, distribution=dist, compile=True)
        te = eager.fit_transform(x)
        tc = comp.fit_transform(x)
        assert torch.allclose(te, tc, atol=1e-6, equal_nan=True), f"compiled transform != eager for {dist}"
        ie = eager.inverse_transform(te)
        ic = comp.inverse_transform(tc)
        assert torch.allclose(ie, ic, atol=1e-6, equal_nan=True), f"compiled inverse != eager for {dist}"


@pytest.mark.parametrize("scaler_cls", [DStandardScalerTensor, DMinMaxScalerTensor])
@pytest.mark.parametrize("channels_last", [False, True])
@pytest.mark.parametrize("perm", [[1, 0, 2, 3], [1, 2, 0, 3], [3, 0, 1, 2], [2, 3, 1, 0]])
def test_tensor_scaler_refit_reordered_columns(scaler_cls, channels_last, perm):
    """Refitting on data whose channels arrive in a different order must match one fit on all the data.

    Covers a single swap and longer cycles: a swap is its own inverse, so it hid a bug where refit indexed the
    input with the input->scaler map instead of its inverse and mixed up columns for any 3+ cycle.
    """
    torch.manual_seed(11)
    names = ["a", "b", "c", "d"]
    scale = torch.tensor([1.0, 10.0, 100.0, 1000.0])
    offset = torch.tensor([0.0, 5.0, 50.0, -500.0])

    def make(n):
        x = torch.randn(n, 4, 3, 5, dtype=torch.float64) * scale.view(1, 4, 1, 1) + offset.view(1, 4, 1, 1)
        return x.movedim(1, -1) if channels_last else x

    def named(x, cols):
        x = x.clone()
        x.variable_names = cols
        return x

    channel_dim = -1 if channels_last else 1
    x1, x2 = make(200), make(150)
    x2_perm = x2.index_select(channel_dim, torch.tensor(perm))

    refit = scaler_cls(channels_last=channels_last)
    refit.fit(named(x1, names))
    refit.fit(named(x2_perm, [names[i] for i in perm]))

    reference = scaler_cls(channels_last=channels_last)
    reference.fit(named(torch.cat([x1, x2]), names))

    for got, expected in zip(refit.get_scales(), reference.get_scales()):
        assert torch.allclose(got, expected, rtol=1e-10), f"refit stats differ for perm {perm}"
    x_test = named(x2, names)
    assert torch.allclose(refit.transform(x_test), reference.transform(x_test), rtol=1e-10)


# ---------------------------------------------------------------------------------------------------------------
# DSpatialStandardScalerTensor (per-grid-point standardization)
# ---------------------------------------------------------------------------------------------------------------

def _named(x, names):
    """Attach variable_names to a tensor (slicing/cat drop the attribute) and return it."""
    x.variable_names = list(names)
    return x


def make_spatial_test_data():
    """Three chunks per layout with a spatially varying mean and std, so per-location stats are distinct."""
    gen = torch.Generator().manual_seed(8675309)
    n_examples = [40, 25, 7]
    h, w, n_lev = 6, 9, 4
    chans = ["t", "q", "ps"]
    loc = torch.linspace(-3, 3, h * w, dtype=torch.float64).reshape(h, w)
    base_mean = torch.tensor([280.0, 1e-3, 98000.0], dtype=torch.float64)
    base_sd = torch.tensor([10.0, 1e-4, 1500.0], dtype=torch.float64)
    mean_hw = base_mean[:, None, None] * (1 + 0.05 * loc)          # (C, H, W)
    sd_hw = base_sd[:, None, None] * (1.5 + torch.sin(loc))        # (C, H, W), > 0
    lev = torch.arange(1, n_lev + 1, dtype=torch.float64)[:, None, None]
    data = {"chans": chans, "2d": [], "3d": [], "credit": [], "h": h, "w": w, "n_lev": n_lev}
    for n in n_examples:
        # 2-D spatial: (B, C, H, W)
        data["2d"].append(torch.randn(n, 3, h, w, generator=gen, dtype=torch.float64) * sd_hw + mean_hw)
        # 3-D spatial: (B, C, L, H, W)
        data["3d"].append(torch.randn(n, 3, n_lev, h, w, generator=gen, dtype=torch.float64)
                          * (sd_hw[:, None] * lev) + mean_hw[:, None] * lev)
        # CREDIT layout: (B, L, T, H, W) with the level/variable axis as the channel (dim 1), T=2
        data["credit"].append(torch.randn(n, n_lev, 2, h, w, generator=gen, dtype=torch.float64)
                              * (sd_hw[0] * lev)[:, None] + (mean_hw[0] * lev)[:, None])
    return data


spatial_test_data = make_spatial_test_data()

_SPATIAL_CASES = [
    # (data key, spatial_dims, reduced dims of the concatenated tensor, channel names)
    ("2d", (-2, -1), (0,), ["t", "q", "ps"]),
    ("3d", (-3, -2, -1), (0,), ["t", "q", "ps"]),
    ("credit", (-2, -1), (0, 2), [f"lev{i}" for i in range(spatial_test_data["n_lev"])]),
]


@pytest.mark.parametrize("key,spatial_dims,reduce_dims,names", _SPATIAL_CASES)
def test_dspatial_standard_scaler_stats(key, spatial_dims, reduce_dims, names):
    chunks = spatial_test_data[key]
    all_x = torch.cat(chunks, dim=0)
    true_mean = all_x.mean(dim=reduce_dims)
    true_var = all_x.var(dim=reduce_dims, unbiased=False)

    # 1. Separate scalers per chunk merged with +.
    scalers = []
    for c in chunks:
        s = DSpatialStandardScalerTensor(channels_last=False, spatial_dims=spatial_dims)
        s.fit(_named(c.clone(), names))
        scalers.append(s)
    total = scalers[0] + scalers[1] + scalers[2]
    # 2. Sequential fits on one scaler.
    seq = DSpatialStandardScalerTensor(channels_last=False, spatial_dims=spatial_dims)
    for c in chunks:
        seq.fit(_named(c.clone(), names))

    for s in (total, seq):
        mean, var = s.get_scales()
        assert mean.shape == true_mean.shape and var.shape == true_var.shape == s.n_.shape
        assert list(s.spatial_shape_) == list(true_mean.shape[1:])
        torch.testing.assert_close(mean, true_mean, rtol=1e-5, atol=0)
        torch.testing.assert_close(var, true_var, rtol=1e-5, atol=0)
        n_expected = all_x.numel() / true_mean.numel()
        assert torch.all(s.n_ == n_expected)
    # Stats are genuinely per location (not collapsed to one per channel).
    assert torch.unique(total.mean_x_[0]).numel() > 1

    # Round trip and normalized output statistics.
    x = _named(all_x.clone(), names)
    xt = total.transform(x)
    torch.testing.assert_close(xt.mean(dim=reduce_dims), torch.zeros_like(true_mean), atol=1e-8, rtol=0)
    torch.testing.assert_close(xt.var(dim=reduce_dims, unbiased=False), torch.ones_like(true_var),
                               rtol=1e-6, atol=0)
    torch.testing.assert_close(total.inverse_transform(xt), all_x, rtol=1e-10, atol=1e-10)


def test_dspatial_standard_scaler_nan_handling():
    torch.manual_seed(1)
    x = torch.randn(50, 2, 4, 5, dtype=torch.float64) * 3 + 10
    x[:20, 0, 1, 2] = float("nan")       # fixed location, partially missing
    x[:, 1, 3, 4] = float("nan")         # fixed location, missing everywhere
    x[7, 0, 0, 0] = float("nan")         # one random missing value
    _named(x, ["a", "b"])
    s = DSpatialStandardScalerTensor()
    s.fit(x)

    n_expected = torch.full((2, 4, 5), 50.0, dtype=torch.float64)
    n_expected[0, 1, 2] = 30
    n_expected[1, 3, 4] = 0
    n_expected[0, 0, 0] = 49
    assert torch.equal(s.n_, n_expected)

    # Partially missing location matches stats of the stripped values.
    vals = x[20:, 0, 1, 2]
    assert torch.isclose(s.mean_x_[0, 1, 2], vals.mean())
    assert torch.isclose(s.var_x_[0, 1, 2], vals.var(unbiased=False))
    # All-NaN location: NaN stats, no crash; other locations finite.
    assert torch.isnan(s.mean_x_[1, 3, 4]) and torch.isnan(s.var_x_[1, 3, 4])
    finite_mask = n_expected > 0
    assert torch.isfinite(s.mean_x_[finite_mask]).all() and torch.isfinite(s.var_x_[finite_mask]).all()

    # NaNs pass through transform; everything else is finite. The all-NaN location transforms to NaN.
    xt = s.transform(x)
    assert torch.equal(torch.isnan(xt), torch.isnan(x))
    clean = _named(torch.ones(3, 2, 4, 5, dtype=torch.float64), ["a", "b"])
    out = s.transform(clean)
    assert torch.isnan(out[:, 1, 3, 4]).all()
    out[:, 1, 3, 4] = 0
    assert torch.isfinite(out).all()

    # Merging a chunk that is all-NaN at a location with one that has data must not poison that location.
    y = torch.randn(10, 2, 4, 5, dtype=torch.float64)
    s2 = DSpatialStandardScalerTensor()
    s2.fit(_named(y.clone(), ["a", "b"]))
    merged = s + s2
    assert torch.allclose(merged.mean_x_[1, 3, 4], y[:, 1, 3, 4].mean())
    assert merged.n_[1, 3, 4] == 10


def test_dspatial_standard_scaler_layouts_and_reordering():
    torch.manual_seed(2)
    names = ["c0", "c1", "c2"]
    x_cf = _named(torch.randn(12, 3, 2, 5, 6, dtype=torch.float64) * 4 + 1, names)  # (B, C, T, H, W)
    s_cf = DSpatialStandardScalerTensor(channels_last=False, spatial_dims=(-2, -1))
    s_cf.fit(x_cf)

    # Same data channels-last: (B, T, H, W, C) with spatial axes (-3, -2).
    x_cl = _named(x_cf.permute(0, 2, 3, 4, 1).contiguous(), names)
    s_cl = DSpatialStandardScalerTensor(channels_last=True, spatial_dims=(-3, -2))
    s_cl.fit(x_cl)
    torch.testing.assert_close(s_cl.mean_x_, s_cf.mean_x_)
    torch.testing.assert_close(s_cl.var_x_, s_cf.var_x_)
    torch.testing.assert_close(s_cl.transform(x_cl).permute(0, 4, 1, 2, 3), s_cf.transform(x_cf))

    # channels_last override at transform time: spatial_dims keep their meaning in the scaler's layout.
    out_override = s_cf.transform(x_cl, channels_last=True)
    assert out_override.shape == x_cl.shape
    torch.testing.assert_close(out_override.permute(0, 4, 1, 2, 3), s_cf.transform(x_cf))
    torch.testing.assert_close(s_cf.inverse_transform(out_override, channels_last=True), x_cl)

    # Channel reordering (3-cycle) and subsets via variable_names.
    perm = [2, 0, 1]
    x_perm = _named(x_cf[:, perm].clone(), [names[i] for i in perm])
    torch.testing.assert_close(s_cf.transform(x_perm), s_cf.transform(x_cf)[:, perm])
    x_sub = _named(x_cf[:, [2]].clone(), ["c2"])
    torch.testing.assert_close(s_cf.transform(x_sub), s_cf.transform(x_cf)[:, [2]])
    assert s_cf.transform(x_sub).variable_names == ["c2"]

    # Refit with permuted channel order gives the same stats as refitting in the original order.
    s_a = copy.deepcopy(s_cf)
    s_a.fit(_named(x_cf.clone() * 2, names))
    s_b = copy.deepcopy(s_cf)
    s_b.fit(_named((x_cf * 2)[:, perm].clone(), [names[i] for i in perm]))
    torch.testing.assert_close(s_a.mean_x_, s_b.mean_x_)
    torch.testing.assert_close(s_a.var_x_, s_b.var_x_)
    torch.testing.assert_close(s_a.n_, s_b.n_)

    # subset_columns
    sub = s_cf.subset_columns(["c2", "c0"])
    assert sub.x_columns_ == ["c2", "c0"] and sub.mean_x_.shape == (2, 5, 6)
    torch.testing.assert_close(sub.mean_x_, s_cf.mean_x_[[2, 0]])
    x_sub2 = _named(x_cf[:, [2, 0]].clone(), ["c2", "c0"])
    torch.testing.assert_close(sub.transform(x_sub2), s_cf.transform(x_sub2))
    with pytest.raises(NotImplementedError):
        s_cf.add_variables(sub)


def test_dspatial_standard_scaler_errors():
    names = ["a", "b"]
    x = _named(torch.randn(5, 2, 4, 6), names)
    s = DSpatialStandardScalerTensor()
    s.fit(x)
    wrong = _named(torch.randn(5, 2, 4, 7), names)
    with pytest.raises(ValueError, match="spatial shape"):
        s.fit(wrong)
    with pytest.raises(ValueError, match="spatial shape"):
        s.transform(wrong)
    with pytest.raises(ValueError, match="spatial shape"):
        s.inverse_transform(wrong)
    s_wrong = DSpatialStandardScalerTensor()
    s_wrong.fit(wrong)
    with pytest.raises(ValueError, match="spatial shapes do not match"):
        s + s_wrong
    with pytest.raises(AssertionError, match="not DSpatialStandardScalerTensor"):
        s + DStandardScalerTensor(channels_last=False)
    with pytest.raises(ValueError, match="channels"):
        s.fit(_named(torch.randn(5, 3, 4, 6), ["a", "b", "c"]))

    # Invalid spatial_dims: channel axis, repeated axis, out of range.
    for bad in [(1, -1), (-1, 3), (-2, 4), (-5, -1)]:
        with pytest.raises(ValueError, match="spatial_dims"):
            DSpatialStandardScalerTensor(spatial_dims=bad).fit(x)
    with pytest.raises(ValueError, match="channel axis"):
        DSpatialStandardScalerTensor(channels_last=True, spatial_dims=(-2, -1)).fit(
            _named(torch.randn(5, 4, 6, 2), names))


def test_dspatial_standard_scaler_serialization(tmp_path):
    torch.manual_seed(3)
    x = _named(torch.randn(10, 2, 3, 4, 5, dtype=torch.float64) * 7 + 2, ["u", "v"])
    s = DSpatialStandardScalerTensor(channels_last=False, spatial_dims=(-3, -2, -1))
    s.fit(x)
    expected = s.transform(x)

    path = tmp_path / "spatial_scaler.json"
    save_scaler(s, path)
    loaded = load_scaler(path)
    dict_path = tmp_path / "spatial_scaler_dict.json"
    save_scaler_dict({"input": {"grid": {"uv": s}}}, dict_path)
    loaded_dict = load_scaler_dict(dict_path)["input"]["grid"]["uv"]

    for ld in (loaded, loaded_dict):
        assert type(ld) is DSpatialStandardScalerTensor
        assert ld.is_fit() and ld.channels_last is False and ld.x_columns_ == ["u", "v"]
        for attr in ("mean_x_", "var_x_", "n_"):
            assert torch.equal(getattr(ld, attr), getattr(s, attr))
        torch.testing.assert_close(ld.transform(x), expected, rtol=0, atol=0)
        torch.testing.assert_close(ld.inverse_transform(expected), s.inverse_transform(expected), rtol=0, atol=0)
        # Loaded scalers keep working for refits and merges.
        ld2 = ld + s
        torch.testing.assert_close(ld2.mean_x_, s.mean_x_)
        ld.fit(x)
        torch.testing.assert_close(ld.var_x_, s.var_x_)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dspatial_standard_scaler_dtypes(dtype):
    torch.manual_seed(4)
    x = _named((torch.randn(30, 2, 6, 8) * 50 + 1000).to(dtype), ["a", "b"])
    s = DSpatialStandardScalerTensor()
    s.fit(x)
    assert s.mean_x_.dtype == dtype
    out = s.transform(x)
    assert out.dtype == dtype
    tol = 1e-4 if dtype == torch.float32 else 1e-10
    torch.testing.assert_close(s.inverse_transform(out), x, rtol=tol, atol=tol * 1000)
    # Stats of one dtype applied to input of another: output follows the input dtype.
    other = torch.float64 if dtype == torch.float32 else torch.float32
    assert s.transform(_named(x.to(other), ["a", "b"])).dtype == other


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_dspatial_standard_scaler_cuda():
    torch.manual_seed(5)
    x_cpu = torch.randn(16, 2, 6, 8)
    x_gpu = _named(x_cpu.to("cuda"), ["a", "b"])
    s = DSpatialStandardScalerTensor()
    s.fit(x_gpu)
    assert s.mean_x_.is_cuda
    out = s.transform(x_gpu)
    assert out.is_cuda
    out_cpu = s.transform(_named(x_cpu.clone(), ["a", "b"]))
    assert not out_cpu.is_cuda
    torch.testing.assert_close(out.cpu(), out_cpu)
    torch.testing.assert_close(s.inverse_transform(out), x_gpu)


def test_dspatial_standard_scaler_matches_flattened_standard_scaler():
    """The new class reproduces what CREDIT does today: a DStandardScalerTensor on the flattened (B, H*W) grid."""
    torch.manual_seed(6)
    b, h, w = 40, 12, 16
    x = torch.randn(b, 1, 1, h, w, dtype=torch.float64) * torch.linspace(1, 5, h * w).reshape(h, w) + 1000
    names = ["PS"]
    spatial = DSpatialStandardScalerTensor(channels_last=False, spatial_dims=(-2, -1))
    spatial.fit(_named(x.clone(), names))
    flat = DStandardScalerTensor(channels_last=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        flat.fit(x.reshape(b, h * w))
        expected = flat.transform(x.reshape(b, h * w)).reshape(b, 1, 1, h, w)
    torch.testing.assert_close(spatial.mean_x_.reshape(-1), flat.mean_x_)
    torch.testing.assert_close(spatial.var_x_.reshape(-1), flat.var_x_)
    torch.testing.assert_close(spatial.transform(_named(x.clone(), names)), expected, rtol=1e-12, atol=1e-12)


def test_dspatial_standard_scaler_zero_variance():
    torch.manual_seed(7)
    x = torch.randn(20, 1, 3, 4, dtype=torch.float64) + 5
    x[:, 0, 1, 1] = 3.25  # constant location
    s = DSpatialStandardScalerTensor()
    s.fit(_named(x.clone(), ["a"]))
    assert s.var_x_[0, 1, 1] == 0
    probe = _named(torch.full((2, 1, 3, 4), 4.0, dtype=torch.float64), ["a"])
    out = s.transform(probe)
    assert torch.isfinite(out).all()
    assert torch.all(out[:, 0, 1, 1] == 4.0 - 3.25)
    torch.testing.assert_close(s.inverse_transform(out), probe)


def test_dspatial_standard_scaler_speed():
    """Informational timing (not asserted): new class vs CREDIT's flattened DStandardScalerTensor."""
    torch.manual_seed(8)
    h, w = 192, 288
    fit_x = torch.randn(8, 1, 1, h, w) * 1000 + 98000
    x = torch.randn(1, 1, 1, h, w) * 1000 + 98000
    spatial = DSpatialStandardScalerTensor(channels_last=False, spatial_dims=(-2, -1))
    spatial.fit(_named(fit_x.clone(), ["PS"]))
    flat = DStandardScalerTensor(channels_last=False)
    x_flat = x.reshape(1, h * w)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        flat.fit(fit_x.reshape(8, h * w))
        x_named = _named(x.clone(), ["PS"])

        def bench(fn, reps=50):
            fn()
            t0 = time.perf_counter()
            for _ in range(reps):
                fn()
            return (time.perf_counter() - t0) / reps * 1e3

        t_spatial = bench(lambda: spatial.transform(x_named))
        t_flat = bench(lambda: flat.transform(x_flat))
    print(f"\ntransform (1,1,1,{h},{w}): DSpatialStandardScalerTensor {t_spatial:.3f} ms, "
          f"flattened DStandardScalerTensor {t_flat:.3f} ms ({t_flat / t_spatial:.1f}x)")
