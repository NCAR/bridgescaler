"""
Pure-numba port of the merging t-digest fitting code from crick (https://github.com/dask/crick, v0.0.8),
which is itself a C port of Ted Dunning's MergingDigest (https://github.com/tdunning/t-digest).

Only the parts bridgescaler needs are ported: building a digest from data (``update``), merging digests
(``merge``) and reading/restoring the digest state. Every kernel mirrors the corresponding function in crick's
``tdigest_stubs.c`` line for line, including the floating-point operation order, the stable bottom-up merge
sort used to order the buffer, and crick's handling of ties, so the fitted centroids, size, min and max are
bit-for-bit identical to crick's. Do not "simplify" the arithmetic below without re-running the 0 ULP tests in
``bridgescaler/tests/tdigest_test.py``.

crick is distributed under the BSD 3-Clause license:

    Copyright (c) 2016, Jim Crist-Harif. 2023, Dask Developers.
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright notice, this
       list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above copyright notice,
       this list of conditions and the following disclaimer in the documentation
       and/or other materials provided with the distribution.

    3. Neither the name of the copyright holder nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import math

import numpy as np
from numba import njit, prange

CENTROID_DTYPE = np.dtype([('mean', np.float64), ('weight', np.float64)])

DBL_MAX = np.finfo(np.float64).max
DBL_EPSILON = np.finfo(np.float64).eps
# NPY_PI_2 and NPY_PI; halving the double nearest pi is exact, so this is also the double nearest pi/2.
PI = math.pi
PI_2 = math.pi / 2

# Indices into the float/int state vectors shared by the kernels (the scalar fields of crick's tdigest_t).
F_COMPRESSION, F_MIN, F_MAX, F_TOTAL_WEIGHT, F_BUFFER_TOTAL_WEIGHT = range(5)
I_NCENTROIDS, I_BUFFER_NCENTROIDS = range(2)


# Portable asin (fdlibm e_asin.c, via musl). crick calls the platform libm asin, which differs by 1 ULP between
# implementations; using our own asin built from basic IEEE operations makes the numba and torch fits
# (bridgescaler.tdigest_tensor) bit-identical on every platform and device.
#
#   Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
#   Developed at SunSoft, a Sun Microsystems, Inc. business.
#   Permission to use, copy, modify, and distribute this software is freely granted, provided that this notice
#   is preserved.
ASIN_PIO2_HI = 1.57079632679489655800e+00
ASIN_PIO2_LO = 6.12323399573676603587e-17
ASIN_PS = (1.66666666666666657415e-01, -3.25565818622400915405e-01, 2.01212532134862925881e-01,
           -4.00555345006794114027e-02, 7.91534994289814532176e-04, 3.47933107596021167570e-05)
ASIN_QS = (-2.40339491173441421878e+00, 2.02094576023350569471e+00, -6.88283971605453293030e-01,
           7.70381505559019352791e-02)
# High-word branch thresholds of e_asin.c as doubles: 2^-26, smallest normal, and the 0x3fef3333 boundary.
ASIN_TINY = 2.0 ** -26
ASIN_MIN_NORMAL = np.finfo(np.float64).tiny
ASIN_NEAR_ONE = float(np.array([0x3fef333300000000], dtype=np.uint64).view(np.float64)[0])
_PS0, _PS1, _PS2, _PS3, _PS4, _PS5 = ASIN_PS
_QS1, _QS2, _QS3, _QS4 = ASIN_QS


@njit(cache=True)
def _asin_r(z):
    p = z * (_PS0 + z * (_PS1 + z * (_PS2 + z * (_PS3 + z * (_PS4 + z * _PS5)))))
    q = 1.0 + z * (_QS1 + z * (_QS2 + z * (_QS3 + z * _QS4)))
    return p / q


@njit(cache=True)
def _asin(x):
    ax = abs(x)
    if not ax < 1.0:
        if ax == 1.0:
            return x * ASIN_PIO2_HI + 2.0 ** -120
        return np.nan
    if ax < 0.5:
        if ax < ASIN_TINY and ax >= ASIN_MIN_NORMAL:
            return x
        return x + x * _asin_r(x * x)
    z = (1.0 - ax) * 0.5
    s = math.sqrt(z)
    r = _asin_r(z)
    if ax >= ASIN_NEAR_ONE:
        y = ASIN_PIO2_HI - (2.0 * (s + s * r) - ASIN_PIO2_LO)
    else:
        # f = s with its low 32 bits cleared (21 significant bits), computed exactly without bit casts
        m, e = math.frexp(s)
        f = math.ldexp(math.floor(math.ldexp(m, 21)), e - 21)
        c = (z - f * f) / (s + f)
        y = 0.5 * ASIN_PIO2_HI - (2.0 * s * r - (ASIN_PIO2_LO - 2.0 * c) - (0.5 * ASIN_PIO2_HI - 2.0 * f))
    if x < 0:
        return -y
    return y


@njit(cache=True)
def _integrate(c, q):
    if q > 1:
        q = 1.0
    return c * (_asin(2 * q - 1) + PI_2) / PI


@njit(cache=True)
def _sort_pass(n, shift, a_mean, a_weight, b_mean, b_weight):
    """One pass of crick's bottom-up merge sort, reading from ``a`` and writing to ``b``."""
    if shift == 0:
        p = 0
        for i in range(0, n, 2):
            if i == n - 1:
                b_mean[p] = a_mean[i]
                b_weight[p] = a_weight[i]
                p += 1
            elif a_mean[i + 1] < a_mean[i]:
                b_mean[p] = a_mean[i + 1]
                b_weight[p] = a_weight[i + 1]
                b_mean[p + 1] = a_mean[i]
                b_weight[p + 1] = a_weight[i]
                p += 2
            else:
                b_mean[p] = a_mean[i]
                b_weight[p] = a_weight[i]
                b_mean[p + 1] = a_mean[i + 1]
                b_weight[p + 1] = a_weight[i + 1]
                p += 2
    else:
        step = 1 << shift
        for i in range(0, n, step << 1):
            if n < i + step:
                ea = n
                eb = 0
            else:
                ea = i + step
                eb = min(n, i + (step << 1))
            j = i
            k = i + step
            p = i
            while j < ea and k < eb:
                if a_mean[k] < a_mean[j]:
                    b_mean[p] = a_mean[k]
                    b_weight[p] = a_weight[k]
                    k += 1
                else:
                    b_mean[p] = a_mean[j]
                    b_weight[p] = a_weight[j]
                    j += 1
                p += 1
            while j < ea:
                b_mean[p] = a_mean[j]
                b_weight[p] = a_weight[j]
                j += 1
                p += 1
            while k < eb:
                b_mean[p] = a_mean[k]
                b_weight[p] = a_weight[k]
                k += 1
                p += 1


@njit(cache=True)
def _centroid_sort(n, mean, weight, buf_mean, buf_weight):
    """Stable sort of the first ``n`` centroids by mean (crick's ``centroid_sort``), in place."""
    curr = 0
    shift = 0
    while (1 << shift) < n:
        if curr == 0:
            _sort_pass(n, shift, mean, weight, buf_mean, buf_weight)
        else:
            _sort_pass(n, shift, buf_mean, buf_weight, mean, weight)
        curr = 1 - curr
        shift += 1
    if curr == 1:
        for i in range(n):
            mean[i] = buf_mean[i]
            weight[i] = buf_weight[i]


@njit(cache=True)
def _centroid_merge(fstate, istate, merge_mean, merge_weight, weight_so_far, k1, u, w):
    k2 = _integrate(fstate[F_COMPRESSION], (weight_so_far + w) / fstate[F_TOTAL_WEIGHT])
    n = istate[I_NCENTROIDS]
    if n == 0:
        n += 1
        istate[I_NCENTROIDS] = n
        merge_weight[n - 1] = w
        merge_mean[n - 1] = u
    elif (k2 - k1) <= 1:
        merge_weight[n - 1] += w
        merge_mean[n - 1] += (u - merge_mean[n - 1]) * w / merge_weight[n - 1]
    else:
        n += 1
        istate[I_NCENTROIDS] = n
        merge_weight[n - 1] = w
        merge_mean[n - 1] = u
        k1 = _integrate(fstate[F_COMPRESSION], weight_so_far / fstate[F_TOTAL_WEIGHT])
    return k1


@njit(cache=True)
def _flush(fstate, istate, c_mean, c_weight, m_mean, m_weight, b_mean, b_weight, s_mean, s_weight):
    """Merge the buffer into the centroids (crick's ``tdigest_flush``)."""
    bn = istate[I_BUFFER_NCENTROIDS]
    if bn == 0:
        return

    _centroid_sort(bn, b_mean, b_weight, s_mean, s_weight)

    if fstate[F_MIN] > b_mean[0]:
        fstate[F_MIN] = b_mean[0]
    if fstate[F_MAX] < b_mean[bn - 1]:
        fstate[F_MAX] = b_mean[bn - 1]

    n = istate[I_NCENTROIDS]
    istate[I_NCENTROIDS] = 0
    fstate[F_TOTAL_WEIGHT] += fstate[F_BUFFER_TOTAL_WEIGHT]
    fstate[F_BUFFER_TOTAL_WEIGHT] = 0.0

    i = 0
    j = 0
    k1 = 0.0
    weight_so_far = 0.0
    while i < bn and j < n:
        if b_mean[i] < c_mean[j]:
            u = b_mean[i]
            w = b_weight[i]
            i += 1
        else:
            u = c_mean[j]
            w = c_weight[j]
            j += 1
        k1 = _centroid_merge(fstate, istate, m_mean, m_weight, weight_so_far, k1, u, w)
        weight_so_far += w
    while i < bn:
        u = b_mean[i]
        w = b_weight[i]
        i += 1
        k1 = _centroid_merge(fstate, istate, m_mean, m_weight, weight_so_far, k1, u, w)
        weight_so_far += w
    while j < n:
        u = c_mean[j]
        w = c_weight[j]
        j += 1
        k1 = _centroid_merge(fstate, istate, m_mean, m_weight, weight_so_far, k1, u, w)
        weight_so_far += w

    istate[I_BUFFER_NCENTROIDS] = 0
    # crick swaps the centroid and merge pointers; copying the merged centroids back is equivalent.
    for k in range(istate[I_NCENTROIDS]):
        c_mean[k] = m_mean[k]
        c_weight[k] = m_weight[k]


@njit(cache=True)
def _add(fstate, istate, c_mean, c_weight, m_mean, m_weight, b_mean, b_weight, s_mean, s_weight, xi, wi):
    """Add one value to the digest (crick's ``tdigest_add``)."""
    # Ignore x = NAN, INF, and -INF, and w <= eps
    if not np.isfinite(xi) or wi <= DBL_EPSILON:
        return
    if istate[I_BUFFER_NCENTROIDS] == b_mean.shape[0]:
        _flush(fstate, istate, c_mean, c_weight, m_mean, m_weight, b_mean, b_weight, s_mean, s_weight)
    n = istate[I_BUFFER_NCENTROIDS]
    istate[I_BUFFER_NCENTROIDS] = n + 1
    b_mean[n] = xi
    b_weight[n] = wi
    fstate[F_BUFFER_TOTAL_WEIGHT] += wi


@njit(cache=True)
def _add_values(fstate, istate, c_mean, c_weight, m_mean, m_weight, b_mean, b_weight, s_mean, s_weight,
                x, w):
    """Add each ``(x[i], w[i])`` to the digest in order."""
    for idx in range(x.shape[0]):
        _add(fstate, istate, c_mean, c_weight, m_mean, m_weight, b_mean, b_weight, s_mean, s_weight, x[idx], w[idx])


@njit(cache=True)
def _add_constant_weight(fstate, istate, c_mean, c_weight, m_mean, m_weight, b_mean, b_weight, s_mean,
                         s_weight, x, w):
    """``_add_values`` for a single scalar weight shared by every value, without materializing a weight array."""
    for idx in range(x.shape[0]):
        _add(fstate, istate, c_mean, c_weight, m_mean, m_weight, b_mean, b_weight, s_mean, s_weight, x[idx], w)


class TDigest(object):
    """
    Merging t-digest approximate histogram, a drop-in replacement for the subset of ``crick.TDigest`` used by
    bridgescaler. Fitting results (centroids, size, min, max) are bit-for-bit identical to crick 0.0.8.

    Args:
        compression (float): Compression factor; larger values keep more centroids. Clipped to [20, 1000].
    """
    def __init__(self, compression=100.0):
        if not np.isfinite(compression):
            raise ValueError("Compression must be finite")
        compression = float(compression)
        if compression < 20:
            compression = 20.0
        elif compression > 1000:
            compression = 1000.0
        size = int(2 * math.ceil(compression))
        buffer_size = int(7.5 + 0.37 * compression - 2e-4 * compression * compression)
        self._fstate = np.array([compression, DBL_MAX, -DBL_MAX, 0.0, 0.0], dtype=np.float64)
        self._istate = np.zeros(2, dtype=np.int64)
        self._c_mean = np.zeros(size)
        self._c_weight = np.zeros(size)
        self._m_mean = np.zeros(size)
        self._m_weight = np.zeros(size)
        self._b_mean = np.zeros(buffer_size)
        self._b_weight = np.zeros(buffer_size)
        self._s_mean = np.zeros(buffer_size)
        self._s_weight = np.zeros(buffer_size)

    def _arrays(self):
        return (self._fstate, self._istate, self._c_mean, self._c_weight, self._m_mean, self._m_weight,
                self._b_mean, self._b_weight, self._s_mean, self._s_weight)

    def _flush(self):
        _flush(*self._arrays())

    def __repr__(self):
        return "TDigest<compression={0}, size={1}>".format(self.compression, self.size())

    @property
    def compression(self):
        """The compression factor for this digest."""
        return self._fstate[F_COMPRESSION]

    def min(self):
        """The minimum value in the digest."""
        self._flush()
        if self._fstate[F_TOTAL_WEIGHT] > 0:
            return self._fstate[F_MIN]
        return np.nan

    def max(self):
        """The maximum value in the digest."""
        self._flush()
        if self._fstate[F_TOTAL_WEIGHT] > 0:
            return self._fstate[F_MAX]
        return np.nan

    def size(self):
        """The sum of the weights on all centroids."""
        return self._fstate[F_TOTAL_WEIGHT] + self._fstate[F_BUFFER_TOTAL_WEIGHT]

    def centroids(self):
        """A structured copy of the digest's centroids with fields ``mean`` and ``weight``."""
        self._flush()
        n = self._istate[I_NCENTROIDS]
        result = np.empty(n, dtype=CENTROID_DTYPE)
        result["mean"] = self._c_mean[:n]
        result["weight"] = self._c_weight[:n]
        return result

    def update(self, x, w=1):
        """
        Add many samples to this digest.

        Args:
            x (array_like): The values to add. Non-finite values are ignored.
            w (array_like): The weight (or weights, broadcast against x) of the values. Default is 1.
        """
        x = np.asarray(x)
        check_w = not (np.isscalar(w) and w == 1)
        w = np.asarray(w)
        if not np.can_cast(x.dtype, np.float64, casting="safe"):
            raise TypeError("x must be numeric")
        if not np.can_cast(w.dtype, np.float64, casting="safe"):
            raise TypeError("w must be numeric")
        if check_w and (not np.isfinite(w).all() or (w <= 0).any()):
            raise ValueError("w must be > 0 and finite")
        if x.size == 0 or w.size == 0:
            return
        # Same iterator setup as crick's tdigest_update_ndarray (NpyIter_MultiNew with NPY_KEEPORDER, buffered,
        # external loop, float64), so values are added in exactly the same order. KEEPORDER flips axes with
        # negative strides, which np.ravel(order="K") does not, so a plain ravel is not equivalent.
        it = np.nditer([x, w], flags=["external_loop", "buffered"], op_flags=[["readonly", "aligned"]] * 2,
                       op_dtypes=[np.float64, np.float64], order="K", casting="safe")
        arrays = self._arrays()
        with it:
            for x_chunk, w_chunk in it:
                if w_chunk.strides[0] == 0:
                    _add_constant_weight(*arrays, x_chunk, float(w_chunk[0]))
                else:
                    _add_values(*arrays, x_chunk, w_chunk)

    def merge(self, *args):
        """
        Update this digest in place with data from other digests.

        Args:
            *args (TDigest): digests to merge into this one.
        """
        if not all(isinstance(i, TDigest) for i in args):
            raise TypeError("All arguments to merge must be TDigests")
        for td in args:
            td._flush()
            if td._fstate[F_TOTAL_WEIGHT]:
                n = td._istate[I_NCENTROIDS]
                # copy first: td may be self, and adding to self can flush and overwrite its centroids
                _add_values(*self._arrays(), td._c_mean[:n].copy(), td._c_weight[:n].copy())
                if self._fstate[F_MIN] > td._fstate[F_MIN]:
                    self._fstate[F_MIN] = td._fstate[F_MIN]
                if self._fstate[F_MAX] < td._fstate[F_MAX]:
                    self._fstate[F_MAX] = td._fstate[F_MAX]

    def __reduce__(self):
        return (TDigest, (self.compression,), self.__getstate__())

    def __getstate__(self):
        centroids = self.centroids()
        return (centroids, self._fstate[F_TOTAL_WEIGHT], self._fstate[F_MIN], self._fstate[F_MAX])

    def __setstate__(self, state):
        """Restore ``(centroids, total_weight, min, max)``. As in crick, the buffer is left untouched."""
        self._fstate[F_TOTAL_WEIGHT] = float(state[1])
        self._fstate[F_MIN] = float(state[2])
        self._fstate[F_MAX] = float(state[3])
        centroids = np.asarray(state[0])
        if centroids.dtype.names is not None:
            means = centroids["mean"]
            weights = centroids["weight"]
        else:
            centroids = centroids.reshape(-1, 2)
            means = centroids[:, 0]
            weights = centroids[:, 1]
        n = means.shape[0]
        if n > 0:
            if n > self._c_mean.shape[0]:
                raise ValueError("More centroids than this digest's compression allows.")
            self._c_mean[:n] = means
            self._c_weight[:n] = weights
            self._istate[I_NCENTROIDS] = n


@njit(parallel=True, cache=True)
def _fit_batch_kernel(x, fstate, istate, c_mean, c_weight, m_mean, m_weight, b_mean, b_weight, s_mean, s_weight):
    """Fit one digest per channel ``i`` of ``x`` (shape ``(A, C, K)``) from the values ``x[:, i, :]`` in C order,
    in parallel over channels. Each channel runs the same sequential kernels as ``TDigest.update`` on its own state
    row, so the result is identical to fitting the channels one at a time."""
    n_a, n_c, n_k = x.shape
    for i in prange(n_c):
        # per-channel row views, created once rather than for every value
        fs, ist, cm, cw, mm, mw = fstate[i], istate[i], c_mean[i], c_weight[i], m_mean[i], m_weight[i]
        bm, bw, sm, sw = b_mean[i], b_weight[i], s_mean[i], s_weight[i]
        for a in range(n_a):
            for k in range(n_k):
                _add(fs, ist, cm, cw, mm, mw, bm, bw, sm, sw, np.float64(x[a, i, k]), 1.0)
        _flush(fs, ist, cm, cw, mm, mw, bm, bw, sm, sw)


@njit(parallel=True, cache=True)
def _merge_batch_kernel(targets, fstate, istate, c_mean, c_weight, m_mean, m_weight, b_mean, b_weight, s_mean,
                        s_weight, o_fstate, o_istate, o_mean, o_weight):
    """Merge flushed digest ``j`` of the other batch into digest ``targets[j]`` (``TDigest.merge``), in parallel.
    ``targets`` must not repeat."""
    for j in prange(targets.shape[0]):
        i = targets[j]
        fs, ist, cm, cw, mm, mw = fstate[i], istate[i], c_mean[i], c_weight[i], m_mean[i], m_weight[i]
        bm, bw, sm, sw = b_mean[i], b_weight[i], s_mean[i], s_weight[i]
        if o_fstate[j, F_TOTAL_WEIGHT] != 0:
            for k in range(o_istate[j, I_NCENTROIDS]):
                _add(fs, ist, cm, cw, mm, mw, bm, bw, sm, sw, o_mean[j, k], o_weight[j, k])
            if fs[F_MIN] > o_fstate[j, F_MIN]:
                fs[F_MIN] = o_fstate[j, F_MIN]
            if fs[F_MAX] < o_fstate[j, F_MAX]:
                fs[F_MAX] = o_fstate[j, F_MAX]
        _flush(fs, ist, cm, cw, mm, mw, bm, bw, sm, sw)


class TDigestArrays(object):
    """
    A batch of ``C`` digests (one per channel) sharing one compression, stored as stacked numpy arrays and fit or
    merged in parallel over channels with numba ``prange``. Results are identical to using one ``TDigest`` per
    channel. The state is kept flushed between calls.

    Args:
        n (int): number of digests.
        compression (float): compression shared by all digests.
    """
    def __init__(self, n, compression=100.0):
        template = TDigest(compression)
        self.compression = template.compression
        self._fstate = np.tile(template._fstate, (n, 1))
        self._istate = np.zeros((n, 2), dtype=np.int64)
        size = template._c_mean.shape[0]
        buffer_size = template._b_mean.shape[0]
        self._c_mean = np.zeros((n, size))
        self._c_weight = np.zeros((n, size))
        self._m_mean = np.zeros((n, size))
        self._m_weight = np.zeros((n, size))
        self._b_mean = np.zeros((n, buffer_size))
        self._b_weight = np.zeros((n, buffer_size))
        self._s_mean = np.zeros((n, buffer_size))
        self._s_weight = np.zeros((n, buffer_size))

    def __len__(self):
        return self._fstate.shape[0]

    def _arrays(self):
        return (self._fstate, self._istate, self._c_mean, self._c_weight, self._m_mean, self._m_weight,
                self._b_mean, self._b_weight, self._s_mean, self._s_weight)

    @staticmethod
    def fit(x, compression=100.0, channel_axis=-1):
        """
        Fit one digest per channel of ``x``. Channel ``i`` sees ``np.take(x, i, channel_axis).ravel()``, in that
        order, exactly as if it were passed to ``TDigest.update``.

        Args:
            x (numpy.ndarray): data with a channel axis.
            compression (float): compression for every digest.
            channel_axis (int): axis holding the channels.
        """
        x = np.asarray(x)
        if not np.can_cast(x.dtype, np.float64, casting="safe"):
            raise TypeError("x must be numeric")
        axis = channel_axis % x.ndim
        n = x.shape[axis]
        # (A, C, K) view in logical C order: values of channel i are x3[:, i, :] read row by row.
        x3 = x.reshape(int(np.prod(x.shape[:axis], dtype=np.int64)), n, -1) if x.size else x.reshape(0, n, 0)
        out = TDigestArrays(n, compression)
        if x.size:
            _fit_batch_kernel(x3, *out._arrays())
        return out

    @staticmethod
    def from_states(centroids, sizes, mins, maxs, compression=100.0):
        """Build from per-channel ``(centroids (k, 2), size, min, max)`` as stored by the scalers (each channel is
        restored like ``TDigest.__setstate__``)."""
        out = TDigestArrays(len(centroids), compression)
        for i, c in enumerate(centroids):
            c = np.asarray(c, dtype=np.float64).reshape(-1, 2)
            if c.shape[0] > out._c_mean.shape[1]:
                raise ValueError("More centroids than this digest's compression allows.")
            out._c_mean[i, :c.shape[0]] = c[:, 0]
            out._c_weight[i, :c.shape[0]] = c[:, 1]
            out._istate[i, I_NCENTROIDS] = c.shape[0]
        out._fstate[:, F_TOTAL_WEIGHT] = np.asarray(sizes, dtype=np.float64)
        out._fstate[:, F_MIN] = np.asarray(mins, dtype=np.float64)
        out._fstate[:, F_MAX] = np.asarray(maxs, dtype=np.float64)
        return out

    def to_states(self):
        """Per-channel ``(centroids list of (k, 2) arrays, size, min, max)``, with NaN min/max for empty digests
        (what ``TDigest.centroids()``, ``size()``, ``min()`` and ``max()`` report)."""
        counts = self._istate[:, I_NCENTROIDS]
        centroids = [np.stack([self._c_mean[i, :counts[i]], self._c_weight[i, :counts[i]]], axis=1)
                     for i in range(len(self))]
        total = self._fstate[:, F_TOTAL_WEIGHT].copy()
        t_min = np.where(total > 0, self._fstate[:, F_MIN], np.nan)
        t_max = np.where(total > 0, self._fstate[:, F_MAX], np.nan)
        return centroids, total, t_min, t_max

    def merge(self, other, targets=None):
        """
        Merge digest ``j`` of ``other`` into digest ``targets[j]`` of this batch in place (``TDigest.merge``).

        Args:
            other (TDigestArrays): digests to merge in; must have the same compression.
            targets (array_like of int): distinct destination indices, one per digest in ``other``. Defaults to
                ``range(len(other))``.
        """
        if other.compression != self.compression:
            raise ValueError("Compressions differ.")
        targets = np.arange(len(other)) if targets is None else np.asarray(targets, dtype=np.int64)
        if targets.shape[0] != len(other) or np.unique(targets).size != targets.size:
            raise ValueError("targets must hold one distinct index per digest in other.")
        _merge_batch_kernel(targets, *self._arrays(), other._fstate, other._istate, other._c_mean, other._c_weight)
        return self
