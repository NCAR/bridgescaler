from bridgescaler import save_scaler, load_scaler
from bridgescaler.distributed_tensor import DStandardScalerTensor, DMinMaxScalerTensor, DQuantileScalerTensor
import copy
import numpy as np
import pytest
import torch
import os

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
    scaler_objs = [DStandardScalerTensor, DMinMaxScalerTensor, DQuantileScalerTensor]
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


def test_dquantile_tensor_compile_matches_eager():
    # compile=True uses torch.compile(fullgraph=True) on the vmapped kernel. fullgraph=True raises on any graph
    # break, so if this runs at all there is no graph break; we also assert it matches the eager path exactly.
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
