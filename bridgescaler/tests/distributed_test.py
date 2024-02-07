from bridgescaler.distributed import DStandardScaler, DQuantileTransformer
import numpy as np


def test_dstandard_scaler():
    np.random.seed(34325)
    means = np.array([0, 5.3, -2.421, 21456.3, 1.e-5])
    sds = np.array([5, 352.2, 1e-4, 20000.3, 5.3e-2])
    n_examples = np.array([30000, 500, 3352, 88])
    datasets_2d = []
    datasets_4d = []
    tile_width = 8
    for n in range(n_examples.size):
        data2d = np.zeros((n_examples[n], means.size))
        data4d = np.zeros((n_examples[n], tile_width, tile_width, means.size))
        for i in range(means.size):
            data2d[:, i] = np.random.normal(loc=means[i], scale=sds[i], size=n_examples[n])
            data4d[..., i] = np.random.normal(loc=means[i], scale=sds[i],size=(n_examples[n], tile_width, tile_width))
        datasets_2d.append(data2d)
        datasets_4d.append(data4d)
    all_ds_2d = np.vstack(datasets_2d)
    dsses_2d = []
    dsses_4d = []
    for n in range(n_examples.size):
        dsses_2d.append(DStandardScaler())
        dsses_2d[-1].fit(datasets_2d[n])
        dsses_4d.append(DStandardScaler())
        dsses_4d[-1].fit(datasets_4d[n])
    dss_total_2d = np.sum(dsses_2d)
    dss_total_4d = np.sum(dsses_4d)
    mean_2d, var_2d = dss_total_2d.get_scales()
    mean_4d, var_4d = dss_total_4d.get_scales()

    assert mean_2d.shape[0] == means.shape[0] and var_2d.shape[0] == sds.shape[0], "Stat shape mismatch"
    assert mean_4d.shape[0] == means.shape[0] and var_4d.shape[0] == sds.shape[0], "Stat shape mismatch"
    assert np.max(np.abs(mean_2d - all_ds_2d.mean(axis=0))) < 1e-8, "significant difference in means"
    assert np.max(np.abs(var_2d - all_ds_2d.var(axis=0, ddof=1))) < 1e-5, "significant difference in variances"

def test_dquantile_scaler_numpy():
    np.random.seed(34325)
    means = np.array([0, 5.3, -2.421, 21456.3, 1.e-5])
    sds = np.array([5, 352.2, 1e-4, 20000.3, 5.3e-2])
    n_examples = np.array([30000, 500, 3352, 88])
    datasets_2d = []
    datasets_4d = []
    tile_width = 8
    for n in range(n_examples.size):
        data2d = np.zeros((n_examples[n], means.size))
        data4d = np.zeros((n_examples[n], tile_width, tile_width, means.size))
        for i in range(means.size):
            data2d[:, i] = np.random.normal(loc=means[i], scale=sds[i], size=n_examples[n])
            data4d[..., i] = np.random.normal(loc=means[i], scale=sds[i], size=(n_examples[n], tile_width, tile_width))
        datasets_2d.append(data2d)
        datasets_4d.append(data4d)
    dsses_2d = []
    dsses_4d = []
    for n in range(n_examples.size):
        dsses_2d.append(DQuantileTransformer())
        dsses_2d[-1].fit(datasets_2d[n])
        dsses_4d.append(DQuantileTransformer())
        dsses_4d[-1].fit(datasets_4d[n])
        ds_2d_transformed = dsses_2d[-1].transform(datasets_2d[n])
        ds_4d_transformed = dsses_4d[-1].transform(datasets_4d[n])
        assert ds_2d_transformed.max() <= 1, "Quantile transform > 1"
        assert ds_4d_transformed.max() <= 1, "Quantile transform > 1"
    return



