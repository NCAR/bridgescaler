from bridgescaler.deep import DeepStandardScaler, DeepMinMaxScaler, DeepQuantileTransformer
from sklearn.preprocessing import QuantileTransformer
from bridgescaler import save_scaler, load_scaler
import numpy as np
from os.path import exists
import os

def test_deep_standard_scaler():
    save_filename = "deep_standard.json"
    try:
        np.random.seed(352680)

        n_ex = 5000
        n_channels = 4
        dim = 32
        means = np.array([1, 5, -4, 2.5], dtype=np.float32)
        sds = np.array([10, 2, 43.4, 32.], dtype=np.float32)
        x = np.zeros((n_ex, dim, dim, n_channels), dtype=np.float32)
        for chan in range(n_channels):
            x[..., chan] = np.random.normal(means[chan], sds[chan], (n_ex, dim, dim))
        dss = DeepStandardScaler()
        dss.fit(x)
        x_transformed = dss.transform(x)
        x_telephone = dss.inverse_transform(x_transformed)
        assert x_transformed.shape == x.shape, "Shape mismatch"
        assert np.mean(np.abs(x_telephone - x)) < 10 * np.finfo(np.float32).eps, "Significant differences"
        save_scaler(dss, save_filename)
        reloaded_scaler = load_scaler(save_filename)
        x_t_r = reloaded_scaler.transform(x)
        assert np.all(x_transformed == x_t_r), "Scaler reloads properly"
    finally:
        if exists(save_filename):
            os.remove(save_filename)
    return


def test_deep_minmax_scaler():
    np.random.seed(352680)
    n_ex = 5000
    n_channels = 4
    dim = 32
    means = np.array([1, 5, -4, 2.5], dtype=np.float32)
    sds = np.array([10, 2, 43.4, 32.], dtype=np.float32)
    x = np.zeros((n_ex, dim, dim, n_channels), dtype=np.float32)
    for chan in range(n_channels):
        x[..., chan] = np.random.normal(means[chan], sds[chan], (n_ex, dim, dim))
    dss = DeepMinMaxScaler()
    dss.fit(x)
    x_transformed = dss.transform(x)
    x_telephone = dss.inverse_transform(x_transformed)
    assert x_transformed.shape == x.shape, "Shape mismatch"
    assert x_transformed.max() <= 1, "Max greater than 1"
    assert x_transformed.min() >= 0, "Min less than 0"
    assert np.mean(np.abs(x_telephone - x)) < 50 * np.finfo(np.float32).eps, "Significant differences"
    return


def test_deep_quantile_transformer():
    import matplotlib.pyplot as plt
    np.random.seed(352680)
    n_ex = 10000
    n_channels = 4
    dim = 32
    means = np.array([1, 5, -4, 2.5], dtype=np.float64)
    sds = np.array([10, 2, 43.4, 32.], dtype=np.float64)
    x = np.zeros((n_ex, dim, dim, n_channels), dtype=np.float64)
    for chan in range(n_channels):
        x[..., chan] = np.random.normal(means[chan], sds[chan], (n_ex, dim, dim))
    dqs = DeepQuantileTransformer(n_quantiles=1000, stochastic=True)
    dqs.fit(x)
    x_transformed = dqs.transform(x)
    x_telephone = dqs.inverse_transform(x_transformed)
    reg_qs = QuantileTransformer(n_quantiles=1000, subsample=dim * dim * n_ex)
    def flatten_to_2D(X):
        return np.reshape(X, newshape=(X.shape[0] * X.shape[1] * X.shape[2], X.shape[-1]))

    x_flat = flatten_to_2D(x)
    x_scaled = reg_qs.fit_transform(x_flat)
    x_tel_2 = np.reshape(reg_qs.inverse_transform(x_scaled), newshape=(x.shape[0], x.shape[1], x.shape[2], x.shape[3]))
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.pcolormesh(x[10, ..., 0], vmin=x[10, ...,0].min(), vmax=x[10, ...,0].max())
    plt.colorbar()
    plt.title("Original")
    plt.subplot(1, 3, 2)
    plt.pcolormesh(x_telephone[10, ..., 0], vmin=x[10, ...,0].min(), vmax=x[10, ...,0].max())
    plt.colorbar()
    plt.title("Telephone")
    plt.subplot(1, 3, 3)
    diff = x_telephone[10, ..., 0] - x[10, ..., 0]
    d_max = np.max(np.abs(diff))
    plt.pcolormesh(diff, vmin=-d_max, vmax=d_max, cmap="RdBu_r")
    plt.colorbar()
    plt.title("Diff")
    plt.savefig("quantile_test.png", dpi=300, bbox_inches="tight")
    plt.close()
    full_diff = np.abs(x - x_telephone).ravel()
    reg_diff = np.abs(x_tel_2 - x).ravel()
    plt.subplot(2, 1, 1)
    plt.hist(full_diff, bins=100)
    plt.subplot(2, 1, 2)
    plt.hist(reg_diff, bins=100)
    plt.gca().set_yscale('log')
    print(np.count_nonzero(full_diff > 0.01) / full_diff.size)
    print(np.count_nonzero(reg_diff > 0.01) / reg_diff.size)

    plt.savefig("hist_diff.png", dpi=300, bbox_inches="tight")
    plt.close()
    for i in range(4):
        plt.plot(dqs.references_, dqs.quantiles_[i] - reg_qs.quantiles_[:, i], color='b')
        #plt.plot(reg_qs.references_, , color='r', ls='--')
    plt.savefig("quantiles.png", dpi=300, bbox_inches="tight")
    assert x_transformed.shape == x.shape, "Shape mismatch"
    assert x_transformed.max() <= 1, "Max greater than 1"
    assert x_transformed.min() >= 0, "Min less than 0"
    #assert np.max(np.abs(x_telephone - x) / x) < 0.1, "Significant differences"
    return