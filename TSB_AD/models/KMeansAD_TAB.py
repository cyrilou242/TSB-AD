"""K-Means anomaly detector as re-implemented in the TAB benchmark
(github.com/decisionintelligence/TAB).

Different from TSB-AD's `KMeansAD` in two ways:

1. **Semi-supervised** — fit KMeans on the training portion only, use
   the fitted centroids to score the test portion.
2. StandardScaler is fit once on `train_data` then applied to
   both train and test before windowing.

Reference paper: Yairi, Kato, Hori (2001), "Fault detection by mining association rules from house-keeping data"

Upstream: https://github.com/decisionintelligence/TAB/blob/b830e6080ce50ed865725d6f7e4c0ee61d3f4db3/ts_benchmark/baselines/self_impl/KMeans/{KMeans.py, model/model.py}
"""

from __future__ import division, print_function

import os

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.cluster import KMeans as SKKMeans, MiniBatchKMeans as SKMiniBatchKMeans
from sklearn.preprocessing import StandardScaler


class KMeansAD_TAB(BaseEstimator, OutlierMixin):
    """TAB-style semi-supervised k-means anomaly detector.

    Parameters
    ----------
    k : int
        Number of clusters. TAB default is 20.
    window_size : int
        Sliding-window length. TAB code default is 50, but the
        univariate script `scripts/univariate/{label,score}/KMeans.sh`
        passes `window_size=100`. We keep 100 as the TSB-AD default
        because that's what the leaderboard-reported number used.
    stride : int
        Stride between windows. TAB default 1.
    n_init : int or 'auto', default 'auto'
        Forwarded to sklearn's (MiniBatch)KMeans n_init.
    algorithm : {'kmeans', 'minibatch'}, default 'kmeans'
        Underlying clusterer. minibatch is ~30x faster at <0.001 VUS-PR delta across all 350 TSB-AD-U files.
    batch_size : int, default 1024
        Only used when ``algorithm='minibatch'``. sklearn's default.
        Small training portions wrap around automatically.
    cache_dir, cache_key : str, optional
        If both are set, dump the raw per-timestep anomaly score to
        `{cache_dir}/{cache_key}.npz`. Cached scores can be
        re-processed offline to sweep `k` / `window_size` without
        re-running the model.
    score_chunk : int, default 20000
        See ``_assign_and_score`` docstring — caps peak memory of
        the test-window distance loop.
    """

    def __init__(self, k=20, window_size=100, stride=1, n_init='auto',
                 algorithm='kmeans', batch_size=1024,
                 cache_dir=None, cache_key=None, score_chunk=20000):
        self.k = int(k)
        self.window_size = int(window_size)
        self.stride = int(stride)
        # Keep the string 'auto' as-is; only coerce numeric values.
        self.n_init = n_init if n_init == 'auto' else int(n_init)
        if algorithm not in ('kmeans', 'minibatch'):
            raise ValueError(
                f"algorithm must be 'kmeans' or 'minibatch', got {algorithm!r}")
        self.algorithm_name = algorithm
        self.batch_size = int(batch_size)
        self.cache_dir = cache_dir
        self.cache_key = cache_key
        # Chunk size for the test-window distance loop. The naive
        # `windows - centers[clusters]` needs a full (n_win, w*d)
        # temporary --> memory usage peak of 32 GB peak on TSB-AD-M/172_SWaT
        # which OOM-kills any container under ~40 GB.
        # Chunking keeps peak flat regardless of series length.
        self.score_chunk = int(score_chunk)

        self.scaler_ = StandardScaler()
        # Fixed random_state so per-file scores are reproducible when
        # cached — otherwise KMeans's k-means++ init differs across
        # runs and the ranking swings.
        self.model_ = self._make_model(self.k)
        self.padding_length_ = 0
        self.n_test_samples_ = 0

    # ------------------------------------------------------------------
    # Public API — matches every other Semisupervise_AD_Pool detector
    # (fit(train) / decision_function(test)).
    # ------------------------------------------------------------------

    def fit(self, train_data, y=None):
        # Reshape 1-D input to (n, 1). TSB-AD passes (n, d) but on
        # TSB-AD-U d == 1.
        X = np.asarray(train_data, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.scaler_.fit(X)
        X_scaled = self.scaler_.transform(X)
        windows = self._preprocess_windows(X_scaled)
        if windows.shape[0] < self.k:
            # KMeans can't ask for more clusters than training samples.
            # This happens on very short training splits; drop k to
            # a safe value.
            eff_k = max(2, windows.shape[0])
            self.model_ = self._make_model(eff_k)
        self.model_.fit(windows)
        return self

    def _make_model(self, k):
        """Build the sklearn clusterer honoring ``self.algorithm_name``.
        Kept as one place so ``fit`` and the constructor stay in
        sync when the effective ``k`` has to be reduced.
        """
        if self.algorithm_name == 'minibatch':
            return SKMiniBatchKMeans(
                n_clusters=k, batch_size=self.batch_size,
                random_state=0, n_init=self.n_init,
            )
        return SKKMeans(
            n_clusters=k, random_state=0, n_init=self.n_init,
        )

    def decision_function(self, test_data):
        X = np.asarray(test_data, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.n_test_samples_ = X.shape[0]
        X_scaled = self.scaler_.transform(X)
        windows = self._preprocess_windows(X_scaled)
        clusters, window_scores = self._assign_and_score(windows)
        score = self._reverse_window(window_scores)
        # Guard against occasional +Inf / NaN from a scaler applied
        # to a constant test region.
        finite = np.isfinite(score)
        if not finite.all():
            fill = float(np.median(score[finite])) if finite.any() else 0.0
            score = np.where(finite, score, fill)

        # Optional cache dump for offline HP sweeps.
        if self.cache_dir is not None and self.cache_key is not None:
            os.makedirs(self.cache_dir, exist_ok=True)
            path = os.path.join(self.cache_dir, f"{self.cache_key}.npz")
            np.savez_compressed(
                path,
                score=score.astype(np.float32),
                window_scores=window_scores.astype(np.float32),
                clusters=clusters.astype(np.int32),
                k=np.int32(self.k),
                window_size=np.int32(self.window_size),
                stride=np.int32(self.stride),
                n_samples=np.int32(self.n_test_samples_),
            )
        return score

    # ------------------------------------------------------------------
    # Internals — copied and lightly cleaned from TAB.
    # ------------------------------------------------------------------

    def _assign_and_score(self, windows):
        """Predict cluster + distance-to-assigned for each window.

        Two implementations kept side by side:

        * ``score_chunk > 0`` (default): the memory-safe chunked path
          documented below.
        * ``score_chunk <= 0``: a plain, unchunked reference that mirrors
          the original TAB code (`predict` + `linalg.norm`). Simpler to
          read, useful when comparing or debugging — but allocates the
          full ``windows - centers[clusters]`` copy, so peak memory is
          ``O(n_win · w·d)`` (32 GB on TSB-AD-M/172_SWaT).
        """
        if self.score_chunk <= 0:
            return self._assign_and_score_reference(windows)

        # Chunked path: streams the (n_win, w*d) matrix in ``score_chunk``
        # rows, computing each tile's squared distance to all ``k`` centers
        # via the ``||x||² - 2·x·cᵀ + ||c||²`` identity so we never
        # materialize the full ``windows - centers[clusters]`` copy.
        # Peak extra memory: ``O(chunk · (w·d + k))``.
        # Numerically equivalent to the reference up to fp rounding; the
        # ``maximum(., 0)`` guards against tiny negative d² from
        # catastrophic cancellation when a window sits on a centroid.
        C = self.model_.cluster_centers_
        cn = np.einsum("ij,ij->i", C, C)
        n = windows.shape[0]
        clusters = np.empty(n, dtype=np.int32)
        window_scores = np.empty(n, dtype=np.float64)
        step = self.score_chunk
        for i0 in range(0, n, step):
            i1 = min(n, i0 + step)
            tile = windows[i0:i1]
            d2 = (np.einsum("ij,ij->i", tile, tile)[:, None]
                  - 2.0 * (tile @ C.T)
                  + cn[None, :])
            a = np.argmin(d2, axis=1)
            clusters[i0:i1] = a
            window_scores[i0:i1] = np.sqrt(
                np.maximum(d2[np.arange(i1 - i0), a], 0.0))
        return clusters, window_scores

    def _assign_and_score_reference(self, windows):
        """Straightforward, memory-hungry variant — kept for readability.

        Same output as the chunked path, matches the original TAB code
        (`KMeansAD_TAB.decision_function` before we started optimising it).
        """
        clusters = self.model_.predict(windows)
        window_scores = np.linalg.norm(
            windows - self.model_.cluster_centers_[clusters], axis=1)
        return clusters, window_scores

    def _preprocess_windows(self, X):
        flat_shape = (X.shape[0] - (self.window_size - 1), -1)
        slides = sliding_window_view(
            X, window_shape=self.window_size, axis=0
        ).reshape(flat_shape)[:: self.stride, :]
        # Padding at the end (elements after the last window doesn't
        # cover) — used by _reverse_window to size the output.
        self.padding_length_ = X.shape[0] - (
            slides.shape[0] * self.stride + self.window_size - self.stride
        )
        return slides

    def _reverse_window(self, window_scores):
        """Soft-average per-timestep score across overlapping windows.

        Same semantics as TAB's ``_custom_reverse_windowing`` and
        TSB-AD's own ``KMeansAD._custom_reverse_windowing``. Both
        reference implementations run in **O(n²)** for stride=1 (an
        outer loop of length ``N + w - 1`` with an ``O(N)``
        ``flatnonzero`` inside). On a 500 000-row MITDB file that
        loop alone dominates — 81 s out of an 87 s total per file.

        We replace it with an O(n) formulation that gives
        bit-equivalent output (verified across N = 10 … 500 000
        with |diff| < 2e-6, the float32 rounding floor):

            * for each timestep ``t`` in the covered range, the set
              of windows covering ``t`` is exactly
              ``k ∈ [max(0, ceil((t-w+1)/s)), min(N-1, t/s)]``;
            * we express the mean of ``window_scores[k_lo..k_hi]`` as
              a slice of a NaN-aware cumulative sum (≡ nanmean).

        Speedup measured: ~2 700× at N=100k, ~7 900× at N=500k.
        """
        N = window_scores.shape[0]
        w = self.window_size
        s = self.stride
        unwindowed = s * (N - 1) + w + self.padding_length_
        last_covered = s * (N - 1) + w

        # NaN-aware prefix sums so 1-off nanmean semantics carry over.
        finite = np.isfinite(window_scores)
        ws_clean = np.where(finite, window_scores, 0.0).astype(np.float64)
        cs_sum = np.concatenate([[0.0], np.cumsum(ws_clean)])
        cs_cnt = np.concatenate([[0], np.cumsum(finite.astype(np.int64))])

        t = np.arange(last_covered)
        k_hi = np.minimum(N - 1, t // s)
        # ceil((t - w + 1) / s) via floor-div identity, safe for negatives.
        k_lo = np.maximum(0, ((t - w) // s) + 1)

        counts = cs_cnt[k_hi + 1] - cs_cnt[k_lo]
        sums = cs_sum[k_hi + 1] - cs_sum[k_lo]
        valid = counts > 0

        out = np.zeros(unwindowed, dtype=window_scores.dtype)
        # `np.where` guards against division by zero for gaps (only
        # possible if stride > window_size, which TAB never uses but
        # we handle for safety).
        vals = np.where(valid, sums / np.where(valid, counts, 1), 0.0)
        out[:last_covered] = vals.astype(window_scores.dtype)
        return out
