# from bridgescaler import save_scaler, load_scaler, print_scaler, read_scaler
from bridgescaler.distributed_tensor import DStandardScalerTensor, DMinMaxScalerTensor
import numpy as np
import torch
import os

def make_test_data():
    np.random.seed(34325)
    test_data = dict()
    col_names = ["a", "b", "c", "d", "e"]
    test_data["means"] = np.array([0, 5.3, -2.421, 21456.3, 1.e-5])
    test_data["sds"] = np.array([5, 352.2, 1e-4, 20000.3, 5.3e-2])
    test_data["n_examples"] = np.array([1000, 500, 88])
    test_data["numpy_2d"] = []
    test_data["numpy_4d"] = []
    test_data["pandas"] = []
    test_data["xarray"] = []
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
        # save_scaler(dsses_2d[-1], "scaler.json")
        # new_scaler = load_scaler("scaler.json")
        # os.remove("scaler.json")
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
        #save_scaler(dsses_2d[-1], "scaler.json")
        #new_scaler = load_scaler("scaler.json")
        #os.remove("scaler.json")
    dss_total_2d = dsses_2d[0] + dsses_2d[1] + dsses_2d[2]
    dss_total_4d = dsses_4d[0] + dsses_4d[1] + dsses_4d[2]
    min_2d, max_2d = dss_total_2d.get_scales()
    min_4d, max_4d = dss_total_4d.get_scales()
    assert torch.max(torch.abs(min_2d - all_ds_2d.min(axis=0).values)) < 1e-8, "significant difference in minimum"
    assert torch.max(torch.abs(max_2d - all_ds_2d.max(axis=0).values)) < 1e-8, "significant difference in maximum"
