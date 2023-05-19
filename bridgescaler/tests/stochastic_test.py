from bridgescaler.stochastic import StochasticQuantileTransformer
import numpy as np

def test_stochastic_quantile_transformer():
    x = np.random.gamma(3, size=10000)
    x[np.random.randint(0, x.size, 3000)] = 0
    x_in = x.reshape(-1, 1)
    qt = StochasticQuantileTransformer()
    x_t = qt.fit_transform(x_in)
    x_u = qt.inverse_transform(x_t)
    assert x_u.shape == x_in.shape
    assert x_t.max() <= 1
    assert x_t.min() >= 0
    return