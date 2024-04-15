from bridgescaler.distributed import DStandardScaler, DMinMaxScaler, DQuantileTransformer, DQuantileScaler
from bridgescaler import save_scaler, load_scaler
import numpy as np
import pandas as pd
import xarray as xr
import os
from multiprocessing import Pool
import psutil
from scipy.special import ndtri
from scipy.stats import norm
from memory_profiler import profile

def make_test_data():
    np.random.seed(34325)
    test_data = dict()
    col_names = ["a", "b", "c", "d", "e"]
    test_data["means"] = np.array([0, 5.3, -2.421, 21456.3, 1.e-5])
    test_data["sds"] = np.array([5, 352.2, 1e-4, 20000.3, 5.3e-2])
    test_data["n_examples"] = np.array([100000, 500, 88])
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
        test_data["pandas"].append(pd.DataFrame(data2d, columns=col_names, index=np.arange(data2d.shape[0])))
        test_data["xarray"].append(xr.DataArray(data4d,
                                                dims=("batch", "y", "x", "variable"),
                                                coords=dict(batch=np.arange(test_data["n_examples"][n]),
                                                            y=np.arange(tile_width),
                                                            x=np.arange(tile_width),
                                                            variable=col_names)))

    return test_data

def eval_dquantile_scaler(test_data):
    np.random.seed(536)
    dsses_2d = []
    dsses_4d = []
    #pool = None
    pool = Pool(8)
    for n in range(test_data["n_examples"].size):
        dsses_2d.append(DQuantileScaler())
        dsses_2d[-1].fit(test_data["numpy_2d"][n])
        dsses_4d.append(DQuantileScaler())
        dsses_4d[-1].fit(test_data["numpy_4d"][n])
        ds_2d_transformed = dsses_2d[-1].transform(test_data["numpy_2d"][n], pool=pool)
        ds_4d_transformed = dsses_4d[-1].transform(test_data["numpy_4d"][n], pool=pool)
        ds_2d_it = dsses_2d[-1].inverse_transform(ds_2d_transformed, pool=pool)
        ds_4d_it = dsses_4d[-1].inverse_transform(ds_4d_transformed, pool=pool)
        assert ds_2d_transformed.max() <= 1, "Quantile transform > 1"
        assert ds_4d_transformed.max() <= 1, "Quantile transform > 1"
        save_scaler(dsses_2d[-1], "scaler.json")
        new_scaler = load_scaler("scaler.json")
        os.remove("scaler.json")
        assert np.nanargmax(np.abs((new_scaler.min_ - dsses_2d[-1].min_))) == 0, \
            "Differences in scaler centroid values after loading"
    pd_dss = DQuantileScaler()
    pd_trans = pd_dss.fit_transform(test_data["pandas"][0], pool=pool)
    pd_inv_trans = pd_dss.inverse_transform(pd_trans, pool=pool)
    sub_cols = ["d", "b"]
    pd_sub_trans = pd_dss.transform(test_data["pandas"][0][sub_cols], pool=pool)
    assert pd_sub_trans.shape[1] == len(sub_cols), "Did not subset properly"
    pd_sub_inv_trans = pd_dss.inverse_transform(pd_sub_trans, pool=pool)
    assert pd_sub_inv_trans.shape[1] == len(sub_cols), "Did not subset properly on inverse."
    assert type(pd_trans) is type(test_data["pandas"][0]), "Pandas DataFrame type not passed through transform"
    assert type(pd_inv_trans) is type(test_data["pandas"][0]), "Pandas DataFrame type not passed through inverse"
    xr_dss = DQuantileScaler(distribution="normal")
    xr_trans = xr_dss.fit_transform(test_data["xarray"][0], pool=pool)
    xr_inv_trans = xr_dss.inverse_transform(xr_trans, pool=pool)
    assert np.all(~np.isnan(xr_trans)), "nans in transform"
    assert np.all(~np.isnan(xr_inv_trans)), "nans in inverse transform"
    assert xr_trans.shape == test_data["xarray"][0].shape, "shape does not match"
    assert xr_inv_trans.shape == test_data["xarray"][0].shape, "shape does not match"

    #assert np.max(np.abs(xr_inv_trans.values - test_data["xarray"][0].values)) < 1e-3, "Differences in transform"
    combined_scaler = np.sum(dsses_2d)
    assert combined_scaler.size_[0] == test_data["n_examples"].sum(), \
        "Summing did not work properly."
    test_data_c_first = test_data["xarray"][0].transpose("batch", "variable", "y", "x").astype("float32")
    xr_dss_first = xr_dss.transform(test_data_c_first, channels_last=False, pool=pool)
    xr_inv_dss_first = xr_dss.inverse_transform(xr_dss_first, channels_last=False, pool=pool)
    assert xr_dss_first.shape == xr_inv_dss_first.shape, "shape does not match"
    xr_dss_f = DQuantileScaler(distribution="normal", channels_last=False)
    xr_dss_f.fit(test_data_c_first)
    scaled_data_quantile_first = xr_dss_f.transform(test_data_c_first, pool=pool)
    assert scaled_data_quantile_first.shape == test_data_c_first.shape
    if pool is not None:
        pool.close()
        pool.join()
    return

@profile
def small_eval(test_data):
    process = psutil.Process()

    # Record initial memory usage

    test_data_c_first = test_data["xarray"][0].transpose("batch", "variable", "y", "x").astype("float32")
    xr_dss_f = DQuantileScaler(distribution="normal", channels_last=False)
    xr_dss_f.fit(test_data_c_first)
    bt_memory = process.memory_info().rss
    initial_memory = process.memory_info().rss
    print(initial_memory/1e6)
    xr_dss_f.distribution = None
    test_data_c_first = xr_dss_f.transform(test_data_c_first)
    test_data_c_sec = ndtri(test_data_c_first)
    output_arr = np.full((1000, 50, 50), 0.5)
    output_arr = norm.ppf(output_arr)
    output_arr = np.full((1000, 50, 50), 0.5)
    output_arr = ndtri(output_arr)
    at_memory = process.memory_info().rss
    print("final mem:", at_memory / 1e6)

    print("mem diff:", (at_memory - bt_memory) / 1e6)
    return test_data_c_first


if __name__ == "__main__":
    from time import  time

    start = time()
    test_data = make_test_data()
    test_data_c_first = test_data["xarray"][0].transpose("batch", "variable", "y", "x").astype("float32")
    print(test_data["xarray"][0])
    test_data_c_first[:] = small_eval(test_data)
    #eval_dquantile_scaler(test_data)
    stop = time()
    print(stop - start)
