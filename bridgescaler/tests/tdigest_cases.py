"""
Shared dataset and fitting-case definitions for the TDigest reproducibility tests.

Each case fits one or more digests with a given ``TDigest`` class (crick's or bridgescaler's port) and returns
the state bridgescaler persists: ``(centroids, size, min, max)``. ``scripts/make_tdigest_fixtures.py`` runs
these with crick to produce the golden file, and ``tdigest_test.py`` runs them with the port and requires
bit-for-bit equality. The fixture stores the generated inputs too, because numpy's distribution samplers are not
bit-identical across platforms (e.g. x86 vs ARM); ``make_data`` is only reproducible on one machine.
"""
import hashlib

import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured, unstructured_to_structured

CENTROID_DTYPE = np.dtype([('mean', np.float64), ('weight', np.float64)])
COMPRESSIONS = [20, 100, 250, 1000]
N = 30000


def _with_nonfinite(rng, n):
    x = rng.normal(size=n)
    idx = rng.choice(n, size=n // 20, replace=False)
    x[idx[0::3]] = np.nan
    x[idx[1::3]] = np.inf
    x[idx[2::3]] = -np.inf
    return x


def _signed_zeros(rng, n):
    return rng.choice(np.array([0.0, -0.0, 1.0, -1.0]), size=n)


DATASETS = {
    "gaussian": lambda rng, n: rng.normal(2.0, 3.0, size=n),
    "gamma_0.5": lambda rng, n: rng.gamma(0.5, 2.0, size=n),
    "gamma_5": lambda rng, n: rng.gamma(5.0, 1.0, size=n),
    "bimodal": lambda rng, n: np.where(rng.random(n) < 0.3, rng.normal(-4, 0.5, n), rng.normal(3, 1.5, n)),
    "trimodal": lambda rng, n: rng.choice(np.array([-10.0, 0.0, 25.0]), size=n) + rng.normal(0, 1, n),
    "lognormal": lambda rng, n: rng.lognormal(0.0, 1.5, size=n),
    "uniform": lambda rng, n: rng.uniform(-1, 1, size=n),
    "exponential": lambda rng, n: rng.exponential(3.0, size=n),
    "student_t3": lambda rng, n: rng.standard_t(3, size=n),
    "cauchy": lambda rng, n: rng.standard_cauchy(size=n),
    "multiscale": lambda rng, n: np.where(rng.random(n) < 0.5, rng.normal(0, 1e-10, n), rng.normal(0, 1e10, n)),
    "integers": lambda rng, n: rng.integers(0, 7, size=n).astype(np.float64),
    "signed_zeros": _signed_zeros,
    "constant": lambda rng, n: np.full(n, 3.25),
    "with_nonfinite": _with_nonfinite,
    "float32": lambda rng, n: rng.normal(size=n).astype(np.float32),
    "sorted": lambda rng, n: np.sort(rng.normal(size=n)),
    "reverse_sorted": lambda rng, n: np.sort(rng.normal(size=n))[::-1].copy(),
    "tiny_1": lambda rng, n: rng.normal(size=1),
    "tiny_2": lambda rng, n: rng.normal(size=2),
    "small_50": lambda rng, n: rng.gamma(2.0, size=50),
}


def make_data(name):
    """Deterministically generate dataset ``name``."""
    seed = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    return DATASETS[name](rng, N)


def unequal_split(x, k, seed=42):
    rng = np.random.default_rng(seed)
    cuts = np.sort(rng.choice(np.arange(1, max(x.size, 2)), size=k - 1, replace=True)) if x.size > 1 else []
    return np.split(x, cuts)


def to_state(td):
    """The state bridgescaler stores for a digest (see ``DQuantileScaler.td_objs_to_attributes``)."""
    return structured_to_unstructured(td.centroids()), td.size(), td.min(), td.max()


def from_state(td_class, compression, state):
    """Rebuild a digest from stored state (see ``DQuantileScaler.attributes_to_td_objs``)."""
    td = td_class(compression)
    td.__setstate__((unstructured_to_structured(state[0], CENTROID_DTYPE), state[1], state[2], state[3]))
    return td


def fit(td_class, compression, x):
    td = td_class(compression)
    td.update(np.ravel(x))
    return td


def case_single(td_class, c, x):
    return to_state(fit(td_class, c, x))


def case_merge(td_class, c, parts):
    """Fit each part separately, then merge all of them into the first in one call."""
    tds = [fit(td_class, c, p) for p in parts]
    tds[0].merge(*tds[1:])
    return to_state(tds[0])


def case_refit(td_class, c, parts):
    """Repeated ``fit`` on an already-fit scaler: rebuild from state, merge a new digest, store state."""
    state = to_state(fit(td_class, c, parts[0]))
    for p in parts[1:]:
        td = from_state(td_class, c, state)
        td.merge(fit(td_class, c, p))
        state = to_state(td)
    return state


def case_add_sequential(td_class, c, parts):
    """Scaler ``__add__`` folded left to right: both operands are rebuilt from state."""
    states = [to_state(fit(td_class, c, p)) for p in parts]
    acc = states[0]
    for s in states[1:]:
        td = from_state(td_class, c, acc)
        td.merge(from_state(td_class, c, s))
        acc = to_state(td)
    return acc


def case_add_tree(td_class, c, parts):
    """Scaler ``__add__`` as a pairwise map-reduce tree."""
    states = [to_state(fit(td_class, c, p)) for p in parts]
    while len(states) > 1:
        nxt = []
        for i in range(0, len(states) - 1, 2):
            td = from_state(td_class, c, states[i])
            td.merge(from_state(td_class, c, states[i + 1]))
            nxt.append(to_state(td))
        if len(states) % 2:
            nxt.append(states[-1])
        states = nxt
    return states[0]


VARIANTS = {
    "single": lambda cls, c, x: case_single(cls, c, x),
    "merge_eq2": lambda cls, c, x: case_merge(cls, c, np.array_split(x, 2)),
    "merge_eq4": lambda cls, c, x: case_merge(cls, c, np.array_split(x, 4)),
    "merge_eq7": lambda cls, c, x: case_merge(cls, c, np.array_split(x, 7)),
    "merge_uneq4": lambda cls, c, x: case_merge(cls, c, unequal_split(x, 4)),
    "refit_eq4": lambda cls, c, x: case_refit(cls, c, np.array_split(x, 4)),
    "refit_uneq7": lambda cls, c, x: case_refit(cls, c, unequal_split(x, 7)),
    "add_seq_eq4": lambda cls, c, x: case_add_sequential(cls, c, np.array_split(x, 4)),
    "add_tree_eq8": lambda cls, c, x: case_add_tree(cls, c, np.array_split(x, 8)),
}


def iter_cases():
    for name in DATASETS:
        for c in COMPRESSIONS:
            for variant in VARIANTS:
                yield name, c, variant


def case_key(name, c, variant):
    return f"{name}|c{c}|{variant}"


def run_case(td_class, name, c, variant, x=None):
    if x is None:
        x = make_data(name)
    return VARIANTS[variant](td_class, c, x)
