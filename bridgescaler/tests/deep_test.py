from bridgescaler.deep import DeepStandardScaler, DeepMinMaxScaler
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