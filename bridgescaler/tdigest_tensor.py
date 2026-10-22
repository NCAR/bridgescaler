"""
Batched PyTorch port of the t-digest fitting code in ``bridgescaler.tdigest``, bit-for-bit (0 ULP) identical to it.

``bridgescaler.tdigest`` (numba) follows crick's sequential algorithm: values go into a buffer, and every
``buffer_size`` values the buffer is sorted and merged into the centroids in one left-to-right pass. Flushes depend
on each other, so they still run one after another here, but each flush is done for every channel (digest) at once
and every step inside a flush is parallel:

* the stable buffer sort is ``torch.sort(stable=True)`` on keys with ``-0.0`` mapped to ``0.0`` (crick compares
  with ``<``, which treats the two zeros as equal; CUDA's radix sort would not);
* crick's two-pointer merge of the sorted buffer with the centroids is reproduced exactly with two
  ``searchsorted`` calls against the running maximum of the centroid means;
* the merge pass decides where a new centroid starts with ``fl(k2 - k1) > 1``. ``k2`` is non-decreasing, so for
  every possible group start ``s`` the next start is found with a vectorized binary search on that exact predicate,
  and the chain of starts reachable from 0 is extracted with binary lifting (pointer doubling);
* the running weighted mean of each centroid is updated in rounds with crick's exact recurrence, all centroids at
  once, so the floating-point operation sequence per centroid is the same as crick's.

Weights must be integers (bridgescaler only fits unit-weight data and merges digests built from it), which makes
every cumulative weight exact regardless of summation order. ``asin`` is the portable fdlibm implementation shared
with the numba code, built from basic IEEE operations. Do not use fused ops (``addcmul``, ``lerp``, ...) or allow
FMA contraction in this module: either would change the rounding.

Performance: because flushes are sequential, the GPU only parallelizes across channels. Compiled with CUDA graphs on
an A100, fitting 64 / 256 channels of 200k values took 0.42 / 0.79 s versus 0.27 / 0.70 s for the numba ``prange``
backend (``bridgescaler.tdigest.TDigestArrays``) on a 32-core CPU node, and each new channel count costs minutes of
compilation. Eager mode is ~150 small kernels plus a device sync per flush. Use this module when the data should
stay on the device; otherwise prefer the numba backend. ``scripts/benchmark_tdigest_fit.py`` measures both.
"""
import math
from dataclasses import dataclass

import numpy as np
import torch

from .tdigest import (ASIN_PIO2_HI, ASIN_PIO2_LO, ASIN_PS, ASIN_QS, ASIN_TINY, ASIN_MIN_NORMAL, ASIN_NEAR_ONE,
                      DBL_MAX, PI, PI_2, F_MIN, F_MAX, F_TOTAL_WEIGHT, I_NCENTROIDS, TDigest)

_LOW_WORD_MASK = -(1 << 32)  # clears the low 32 bits of a float64 viewed as int64


def digest_sizes(compression):
    """Clipped compression, centroid capacity and buffer size, exactly as ``TDigest.__init__`` computes them."""
    compression = float(compression)
    if compression < 20:
        compression = 20.0
    elif compression > 1000:
        compression = 1000.0
    size = int(2 * math.ceil(compression))
    buffer_size = int(7.5 + 0.37 * compression - 2e-4 * compression * compression)
    return compression, size, buffer_size


@dataclass
class TDigestBatch:
    """Flushed state of ``C`` independent t-digests (one per channel) stored as padded tensors.

    Attributes:
        mean, weight: ``(C, size)`` centroids; entries at or beyond ``count`` are 0.
        count: ``(C,)`` int64 number of centroids.
        total_weight, min, max: ``(C,)`` float64. ``min``/``max`` are the raw internal values (``DBL_MAX`` /
            ``-DBL_MAX`` for an empty digest), as in crick.
        compression: the clipped compression shared by all channels.
    """
    mean: torch.Tensor
    weight: torch.Tensor
    count: torch.Tensor
    total_weight: torch.Tensor
    min: torch.Tensor
    max: torch.Tensor
    compression: float

    @staticmethod
    def empty(n, compression, device=None):
        compression, size, _ = digest_sizes(compression)
        f64 = dict(dtype=torch.float64, device=device)
        return TDigestBatch(torch.zeros((n, size), **f64), torch.zeros((n, size), **f64),
                            torch.zeros(n, dtype=torch.int64, device=device), torch.zeros(n, **f64),
                            torch.full((n,), DBL_MAX, **f64), torch.full((n,), -DBL_MAX, **f64), compression)

    @staticmethod
    def from_states(centroids, sizes, mins, maxs, compression, device=None):
        """Build from the per-channel state bridgescaler stores (``centroids_``, ``size_``, ``min_``, ``max_``),
        the equivalent of ``TDigest.__setstate__`` for each channel."""
        out = TDigestBatch.empty(len(centroids), compression)
        for i, c in enumerate(centroids):
            c = np.asarray(c, dtype=np.float64).reshape(-1, 2)
            if c.shape[0] > out.mean.shape[1]:
                raise ValueError("More centroids than this digest's compression allows.")
            out.mean[i, :c.shape[0]] = torch.from_numpy(c[:, 0])
            out.weight[i, :c.shape[0]] = torch.from_numpy(c[:, 1])
            out.count[i] = c.shape[0]
        out.total_weight = torch.as_tensor(np.asarray(sizes, dtype=np.float64)).clone()
        out.min = torch.as_tensor(np.asarray(mins, dtype=np.float64)).clone()
        out.max = torch.as_tensor(np.asarray(maxs, dtype=np.float64)).clone()
        return out.to(device)

    def select(self, index):
        """Channels ``index`` (a list of ints) as a new batch, in that order."""
        idx = torch.as_tensor(index, dtype=torch.long, device=self.mean.device)
        return TDigestBatch(*(t.index_select(0, idx) for t in (self.mean, self.weight, self.count,
                                                              self.total_weight, self.min, self.max)),
                            self.compression)

    def to(self, device):
        return TDigestBatch(*(t.to(device) for t in (self.mean, self.weight, self.count, self.total_weight,
                                                     self.min, self.max)), self.compression)

    def to_states(self):
        """Per-channel ``(centroids (k, 2), size, min(), max())`` numpy state, like ``TDigest`` would report it."""
        mean, weight, count = self.mean.cpu().numpy(), self.weight.cpu().numpy(), self.count.cpu().numpy()
        total = self.total_weight.cpu().numpy()
        t_min = np.where(total > 0, self.min.cpu().numpy(), np.nan)
        t_max = np.where(total > 0, self.max.cpu().numpy(), np.nan)
        centroids = [np.stack([mean[i, :count[i]], weight[i, :count[i]]], axis=1) for i in range(count.size)]
        return centroids, total, t_min, t_max


def _asin_r(z):
    ps0, ps1, ps2, ps3, ps4, ps5 = ASIN_PS
    qs1, qs2, qs3, qs4 = ASIN_QS
    p = z * (ps0 + z * (ps1 + z * (ps2 + z * (ps3 + z * (ps4 + z * ps5)))))
    q = 1.0 + z * (qs1 + z * (qs2 + z * (qs3 + z * qs4)))
    return p / q


def _sqrt(z):
    """Correctly rounded float64 square root on every device.

    Eager ``torch.sqrt`` on x86 CPU builds with MKL goes through MKL VML, which is not correctly rounded (about 1%
    of float64 inputs differ by 1 ULP), so eager CPU tensors use NumPy's hardware sqrt instead. CUDA ``sqrt`` and
    the C++/Triton code Inductor generates are correctly rounded.
    """
    if z.device.type == "cpu" and not torch.compiler.is_compiling():
        return torch.from_numpy(np.sqrt(z.numpy()))
    return torch.sqrt(z)


def asin_tensor(x):
    """Branch-free tensor version of ``bridgescaler.tdigest._asin`` (fdlibm), bit-identical to it."""
    ax = x.abs()
    small = x + x * _asin_r(x * x)
    small = torch.where((ax < ASIN_TINY) & (ax >= ASIN_MIN_NORMAL), x, small)
    z = (1.0 - ax) * 0.5
    s = _sqrt(z)
    r = _asin_r(z)
    near_one = ASIN_PIO2_HI - (2.0 * (s + s * r) - ASIN_PIO2_LO)
    f = (s.view(torch.int64) & _LOW_WORD_MASK).view(torch.float64)
    c = (z - f * f) / (s + f)
    mid = 0.5 * ASIN_PIO2_HI - (2.0 * s * r - (ASIN_PIO2_LO - 2.0 * c) - (0.5 * ASIN_PIO2_HI - 2.0 * f))
    big = torch.where(ax >= ASIN_NEAR_ONE, near_one, mid)
    big = torch.where(x < 0, -big, big)
    out = torch.where(ax < 0.5, small, big)
    out = torch.where(ax == 1.0, x * ASIN_PIO2_HI + 2.0 ** -120, out)
    return torch.where(ax > 1.0, torch.full_like(out, float("nan")), out)


def integrate_tensor(c, q):
    q = torch.where(q > 1, torch.ones_like(q), q)
    return c * (asin_tensor(2 * q - 1) + PI_2) / PI


def _gather(t, idx):
    return torch.gather(t, 1, idx)


def _flush(state, u, w, nb, rounds):
    """Flush one buffer of up to ``B`` values per channel into ``state`` (crick's ``tdigest_flush``).

    Args:
        state (TDigestBatch): current digests.
        u, w (torch.Tensor): ``(C, B)`` buffer values and weights in insertion order; only the first ``nb[c]``
            entries of row ``c`` are used.
        nb (torch.Tensor): ``(C,)`` number of buffered values per channel; channels with 0 are left unchanged.
        rounds (int or None): number of mean-update rounds to run, i.e. an upper bound on the number of merged
            elements in any one centroid. ``None`` computes the exact bound, which costs a device sync.

    Returns:
        (TDigestBatch, torch.Tensor, torch.Tensor): the new state, a ``(C,)`` bool flag for channels whose result
        cannot be trusted (``rounds`` too small, ``k2`` not monotone, or more centroids than the capacity), and the
        largest centroid group length seen per channel.
    """
    C, B = u.shape
    S = state.mean.shape[1]
    M = B + S
    device = u.device
    c = state.compression
    iB = torch.arange(B, device=device)
    iS = torch.arange(S, device=device)
    iM = torch.arange(M, device=device)
    nc = state.count
    active = nb > 0

    # Stable sort of the buffer by mean (crick's merge sort). Keys map -0.0 to 0.0 and padding to +inf.
    b_valid = iB < nb[:, None]
    key = torch.where(b_valid, torch.where(u == 0, torch.zeros_like(u), u), torch.full_like(u, float("inf")))
    key, order = torch.sort(key, dim=1, stable=True)
    b = _gather(u, order)
    bw = torch.where(b_valid, _gather(w, order), torch.zeros_like(w))
    first = b[:, 0]
    last = _gather(b, (nb - 1).clamp(min=0)[:, None])[:, 0]
    new_min = torch.where(active & (state.min > first), first, state.min)
    new_max = torch.where(active & (state.max < last), last, state.max)
    total = state.total_weight + bw.sum(dim=1)

    # Merge positions reproducing crick's two-pointer merge (buffer value taken only if strictly less).
    c_valid = iS < nc[:, None]
    ckey = torch.where(c_valid, torch.where(state.mean == 0, torch.zeros_like(state.mean), state.mean),
                       torch.full_like(state.mean, float("inf")))
    cmax = torch.cummax(ckey, dim=1).values
    taken_c = torch.searchsorted(cmax, key, right=True)   # centroids emitted before buffer value i
    taken_b = torch.searchsorted(key, cmax, right=False)  # buffer values emitted before centroid j
    pos_b = torch.where(b_valid, iB + taken_c, nc[:, None] + iB)
    pos_c = torch.where(c_valid, iS + taken_b, B + iS)
    U = torch.zeros((C, M), dtype=torch.float64, device=device)
    W = torch.zeros((C, M), dtype=torch.float64, device=device)
    U = U.scatter(1, pos_b, b).scatter(1, pos_c, state.mean)
    W = W.scatter(1, pos_b, bw).scatter(1, pos_c, state.weight)
    n_valid = (nb + nc)[:, None]
    m_valid = iM < n_valid

    # k-scale of the cumulative weight after (k2) and before (k1) each element; weights are integers, so exact.
    cum = torch.cumsum(W, dim=1)
    t = torch.where(active, total, torch.ones_like(total))[:, None]
    k2 = integrate_tensor(c, cum / t)
    k1 = torch.where(iM == 0, torch.zeros_like(cum), integrate_tensor(c, (cum - W) / t))
    k2 = torch.where(m_valid, k2, torch.full_like(k2, float("inf")))
    not_monotone = ((k2[:, 1:] < k2[:, :-1]) & m_valid[:, 1:]).any(dim=1)

    # next[s]: first j > s with fl(k2[j] - k1[s]) > 1 (k2 non-decreasing, so binary search), else n_valid.
    lo = iM[None, :] + 1 + torch.zeros_like(nc)[:, None]
    hi = n_valid.expand(C, M).clone()
    lo = torch.minimum(lo, hi)
    for _ in range(max(1, int(M).bit_length())):
        mid = (lo + hi) // 2
        pred = (_gather(k2, mid.clamp(max=M - 1)) - k1) > 1
        move = lo < hi
        hi = torch.where(move & pred, mid, hi)
        lo = torch.where(move & ~pred, mid + 1, lo)
    nxt = torch.cat([lo, torch.full((C, 1), M, dtype=lo.dtype, device=device)], dim=1)
    nxt = torch.where(torch.arange(M + 1, device=device) >= n_valid, torch.arange(M + 1, device=device), nxt)

    # Group starts p_g = next^g(0) for every g at once, by binary lifting.
    p = torch.zeros((C, M), dtype=torch.int64, device=device)
    jump = nxt
    for level in range(max(1, int(M).bit_length())):
        bit = ((iM >> level) & 1).bool()
        p = torch.where(bit, _gather(jump, p), p)
        jump = _gather(jump, jump)
    g_valid = p < n_valid
    starts = torch.where(g_valid, p, torch.zeros_like(p))
    ends = torch.cat([p[:, 1:], n_valid], dim=1).clamp(max=n_valid.expand(C, M))
    length = torch.where(g_valid, ends - starts, torch.zeros_like(p))
    n_groups = g_valid.sum(dim=1)
    max_len = length.max(dim=1).values

    # Running weighted mean of each centroid, one merged element per round (crick's centroid_merge recurrence).
    mean = _gather(U, starts)
    weight = _gather(W, starts)
    if rounds is None:
        rounds = int(max_len.max())
    for r in range(1, rounds):
        has = r < length
        idx = (starts + r).clamp(max=M - 1)
        ur = _gather(U, idx)
        wr = _gather(W, idx)
        new_weight = torch.where(has, weight + wr, weight)
        mean = torch.where(has, mean + (ur - mean) * wr / new_weight, mean)
        weight = new_weight

    keep = g_valid[:, :S] & active[:, None]
    stay = ~active[:, None]
    out = TDigestBatch(
        torch.where(stay, state.mean, torch.where(keep, mean[:, :S], torch.zeros_like(state.mean))),
        torch.where(stay, state.weight, torch.where(keep, weight[:, :S], torch.zeros_like(state.weight))),
        torch.where(active, n_groups, nc),
        torch.where(active, total, state.total_weight),
        new_min, new_max, c)
    bad = active & (not_monotone | (max_len > rounds) | (n_groups > S))
    return out, bad, torch.where(active, max_len, torch.zeros_like(max_len))


def _compact_finite(x):
    """Move each row's finite values to the front, keeping their order (non-finite values are ignored by crick)."""
    finite = torch.isfinite(x)
    counts = finite.sum(dim=1)
    if bool(finite.all()):
        return x, counts
    pos = torch.where(finite, torch.cumsum(finite, dim=1) - 1, torch.full_like(counts[:, None], x.shape[1]))
    out = torch.zeros((x.shape[0], x.shape[1] + 1), dtype=x.dtype, device=x.device)
    out.scatter_(1, pos, x)
    return out[:, :-1], counts


# Inductor options for the compiled flush: no FMA contraction in generated Triton (Inductor disables Triton's
# fp fusion when emulate_precision_casts is set) or C++ kernels, which would change the rounding of a*b+c.
COMPILE_OPTIONS = {"emulate_precision_casts": True, "cpp.enable_floating_point_contract_flag": "off"}
# Static mean-update round bound for compiled flushes. Continuous data needs ~5-11 rounds; ties can need up to
# buffer_size + 1, and channels that exceed the bound are recomputed with numba (same bits). Compile time and
# per-flush cost grow with the bound, and 16 measured as fast as 8 while covering far more tied data.
COMPILED_ROUNDS = 16
_COMPILED = {}


def _flush_block(state, u, w, nb, minmax_before, other_min, other_max, rounds):
    """``F`` consecutive flushes in one call, so a compiled block launches fewer, larger graphs.

    Args:
        state (TDigestBatch): current digests.
        u, w (torch.Tensor): ``(C, F, B)`` buffers for the ``F`` flushes.
        nb (torch.Tensor): ``(C, F)`` buffered value counts (0 = no-op for that channel and flush).
        minmax_before (torch.Tensor): ``(C, F)`` bool; where set, merge ``other_min``/``other_max`` into the
            digest's min/max right before that flush (``TDigest.merge`` does this before its final flush).
        other_min, other_max (torch.Tensor): ``(C,)``.
        rounds (int or None): see ``_flush``.
    """
    C, F, B = u.shape
    bad = torch.zeros(C, dtype=torch.bool, device=u.device)
    max_len = torch.zeros(C, dtype=torch.int64, device=u.device)
    for f in range(F):
        merge_now = minmax_before[:, f]
        state = TDigestBatch(state.mean, state.weight, state.count, state.total_weight,
                             torch.where(merge_now & (state.min > other_min), other_min, state.min),
                             torch.where(merge_now & (state.max < other_max), other_max, state.max),
                             state.compression)
        state, step_bad, step_len = _flush(state, u[:, f].contiguous(), w[:, f].contiguous(), nb[:, f], rounds)
        bad = bad | step_bad
        max_len = torch.maximum(max_len, step_len)
    return state, bad, max_len


def _compiled_block(mode):
    if mode not in (None, "reduce-overhead"):
        raise ValueError(f"Unsupported compile mode {mode!r}; use True or 'reduce-overhead'.")
    if mode not in _COMPILED:
        # torch.compile takes either a mode or options; reduce-overhead is the triton.cudagraphs option
        options = dict(COMPILE_OPTIONS, **({"triton.cudagraphs": True} if mode else {}))
        _COMPILED[mode] = torch.compile(_flush_block, fullgraph=True, dynamic=False, options=options)
    return _COMPILED[mode]


def default_rounds(compression, device, compile):
    """Mean-update rounds per flush: exact (one device sync per flush) unless compiled, where the static
    ``COMPILED_ROUNDS`` bound keeps the graph sync-free."""
    return COMPILED_ROUNDS if compile else None


class _Fitter:
    """Runs blocks of flushes, optionally compiled (and CUDA-graphed); tracks failure flags without syncing."""
    def __init__(self, compile=False, rounds=None, flushes_per_call=1, mode=None):
        self.rounds = rounds
        self.flushes_per_call = flushes_per_call
        self.graphed = mode == "reduce-overhead"
        self.step = _compiled_block(mode) if compile else _flush_block

    def __call__(self, state, u, w, nb, minmax_before, other_min, other_max, bad, max_len):
        if self.graphed:
            torch.compiler.cudagraph_mark_step_begin()
        state, step_bad, step_len = self.step(state, u, w, nb, minmax_before, other_min, other_max, self.rounds)
        if self.graphed:
            # CUDA-graph outputs are overwritten by the next replay; keep our own copies
            state = TDigestBatch(*(t.clone() for t in (state.mean, state.weight, state.count, state.total_weight,
                                                       state.min, state.max)), state.compression)
            step_bad, step_len = step_bad.clone(), step_len.clone()
        return state, bad | step_bad, torch.maximum(max_len, step_len)


def _compile_settings(compile, flushes_per_call, device):
    """Resolve ``compile`` (False, True or a torch.compile mode string) and the flushes per call."""
    mode = None
    if isinstance(compile, str):
        mode, compile = compile, True
    if flushes_per_call == "auto":
        flushes_per_call = FLUSHES_PER_COMPILED_CALL if compile else 1
    return bool(compile), mode, int(flushes_per_call)


FLUSHES_PER_COMPILED_CALL = 4


def _finish(state, bad, fallback):
    """Recompute any flagged channel with the numba implementation (same algorithm, so same bits)."""
    if not bool(bad.any()):
        return state
    idx = torch.nonzero(bad).flatten().tolist()
    fixed = fallback(idx)
    state = state.to("cpu") if state.mean.device.type != "cpu" else state
    for i, (centroids, size, t_min, t_max) in zip(idx, fixed):
        n = centroids.shape[0]
        state.mean[i].zero_()
        state.weight[i].zero_()
        state.mean[i, :n] = torch.from_numpy(np.ascontiguousarray(centroids[:, 0]))
        state.weight[i, :n] = torch.from_numpy(np.ascontiguousarray(centroids[:, 1]))
        state.count[i] = n
        state.total_weight[i] = size
        state.min[i] = t_min
        state.max[i] = t_max
    return state


def _numba_state(td):
    """Raw flushed state of a numba ``TDigest`` (internal min/max, not the NaN-for-empty ``min()``/``max()``)."""
    td._flush()
    n = int(td._istate[I_NCENTROIDS])
    return (np.stack([td._c_mean[:n], td._c_weight[:n]], axis=1), td._fstate[F_TOTAL_WEIGHT],
            td._fstate[F_MIN], td._fstate[F_MAX])


def _numba_digest(batch, i):
    """Rebuild channel ``i`` of a flushed batch as a numba ``TDigest`` with identical internal state."""
    td = TDigest(batch.compression)
    n = int(batch.count[i])
    centroids = np.stack([batch.mean[i, :n].cpu().numpy(), batch.weight[i, :n].cpu().numpy()], axis=1)
    td.__setstate__((centroids, float(batch.total_weight[i]), float(batch.min[i]), float(batch.max[i])))
    return td


def fit_tdigest_batch(x, compression, compile=False, rounds="auto", flushes_per_call="auto"):
    """Fit one t-digest per row of ``x``, bit-identical to ``TDigest(compression).update(row)``.

    Args:
        x (torch.Tensor): ``(C, N)`` values; each row is one channel in insertion order. Non-finite values are
            skipped. float32 input is upcast exactly to float64 per buffer.
        compression (float): t-digest compression.
        compile (bool or str): run blocks of flushes through ``torch.compile(fullgraph=True)`` to fuse their
            kernels; a string is used as the torch.compile ``mode`` (e.g. ``"reduce-overhead"`` for CUDA graphs).
            Eager mode issues ~150 small kernels and a device sync per flush, so it is only practical on CPU.
        rounds (int, None or "auto"): static bound on merged elements per centroid in a flush. ``None`` computes
            it per flush (one device sync each); an int avoids syncs and any channel that exceeds it is recomputed
            with the numba implementation at the end. ``"auto"`` uses ``default_rounds``.
        flushes_per_call (int or "auto"): flushes per (compiled) call; ``"auto"`` is ``FLUSHES_PER_COMPILED_CALL``
            when compiled and 1 otherwise.

    Returns:
        TDigestBatch: the flushed digests on ``x.device``.
    """
    if x.dim() != 2:
        raise ValueError("x must have shape (channels, values)")
    compression, _, B = digest_sizes(compression)
    C, N = x.shape
    device = x.device
    compile, mode, F = _compile_settings(compile, flushes_per_call, device)
    if rounds == "auto":
        rounds = default_rounds(compression, device, compile)
    state = TDigestBatch.empty(C, compression, device)
    xc, counts = _compact_finite(x)
    fitter = _Fitter(compile, rounds, F, mode)
    bad = torch.zeros(C, dtype=torch.bool, device=device)
    max_len = torch.zeros(C, dtype=torch.int64, device=device)
    ones = torch.ones((C, F, B), dtype=torch.float64, device=device)
    no_minmax = torch.zeros((C, F), dtype=torch.bool, device=device)
    dummy = torch.zeros(C, dtype=torch.float64, device=device)
    offsets = torch.arange(F, device=device) * B
    for start in range(0, N, F * B):
        block = xc[:, start:start + F * B].to(torch.float64)
        if block.shape[1] < F * B:
            block = torch.nn.functional.pad(block, (0, F * B - block.shape[1]))
        nb = (counts[:, None] - start - offsets).clamp(min=0, max=B)
        # contiguous so the compiled block sees the same strides for every N (no stride-guard recompiles)
        block = block.reshape(C, F, B).contiguous()
        state, bad, max_len = fitter(state, block, ones, nb, no_minmax, dummy, dummy, bad, max_len)

    def fallback(idx):
        out = []
        xn = x[idx].cpu().numpy()
        for row in xn:
            td = TDigest(compression)
            td.update(row)
            out.append(_numba_state(td))
        return out
    return _finish(state, bad, fallback).to(device)


def merge_tdigest_batch(state, other, compile=False, rounds="auto", flushes_per_call="auto"):
    """Merge ``other`` into ``state`` channel by channel, bit-identical to ``TDigest.merge``.

    Both batches must be flushed (as returned by ``fit_tdigest_batch``/``from_states``) with the same compression.
    ``compile``, ``rounds`` and ``flushes_per_call`` are as in ``fit_tdigest_batch``.
    """
    if state.compression != other.compression:
        raise ValueError("Compressions differ.")
    _, _, B = digest_sizes(state.compression)
    C, S = other.mean.shape
    device = state.mean.device
    other = other.to(device)
    original = state
    compile, mode, F = _compile_settings(compile, flushes_per_call, device)
    if rounds == "auto":
        rounds = default_rounds(state.compression, device, compile)
    fitter = _Fitter(compile, rounds, F, mode)
    bad = torch.zeros(C, dtype=torch.bool, device=device)
    max_len = torch.zeros(C, dtype=torch.int64, device=device)
    # crick adds other's centroids to the buffer, flushing whenever it is full, then merges min/max, and the last
    # (partial or full) buffer is only flushed afterwards when the result is read.
    use = other.total_weight != 0
    n_add = torch.where(use, other.count, torch.zeros_like(other.count))
    last_chunk = torch.div(n_add - 1, B, rounding_mode="floor")
    n_blocks = -(-(-(-S // B)) // F)
    pad = n_blocks * F * B - S
    means = torch.nn.functional.pad(other.mean, (0, pad)).reshape(C, n_blocks, F, B)
    weights = torch.nn.functional.pad(other.weight, (0, pad)).reshape(C, n_blocks, F, B)
    chunk_index = torch.arange(n_blocks * F, device=device).reshape(n_blocks, F)
    for k in range(n_blocks):
        nb = (n_add[:, None] - chunk_index[k] * B).clamp(min=0, max=B)
        minmax_before = use[:, None] & (last_chunk[:, None] == chunk_index[k])
        state, bad, max_len = fitter(state, means[:, k].contiguous(), weights[:, k].contiguous(), nb, minmax_before,
                                     other.min, other.max, bad, max_len)

    def fallback(idx):
        out = []
        for i in idx:
            td = _numba_digest(original, i)
            td.merge(_numba_digest(other, i))
            out.append(_numba_state(td))
        return out
    return _finish(state, bad, fallback).to(device)
