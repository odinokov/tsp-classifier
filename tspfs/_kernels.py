from __future__ import annotations

from typing import Tuple

import numpy as np
from numba import njit, prange


@njit(parallel=True)
def _rank_rows_average_numba(x: np.ndarray) -> np.ndarray:
    n_samples, n_features = x.shape
    ranks = np.empty((n_samples, n_features), dtype=np.float64)

    for sample_idx in prange(n_samples):
        order = np.argsort(x[sample_idx])
        start = 0
        while start < n_features:
            end = start + 1
            value = x[sample_idx, order[start]]
            while end < n_features and x[sample_idx, order[end]] == value:
                end += 1

            avg_rank = 0.5 * ((start + 1) + end)
            for pos in range(start, end):
                ranks[sample_idx, order[pos]] = avg_rank
            start = end

    return ranks


@njit(parallel=True)
def _score_pairs_numba(
    x: np.ndarray,
    ranks: np.ndarray,
    y01: np.ndarray,
    features: np.ndarray,
    n_negative: int,
    n_positive: int,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    n_selected = features.size
    n_pairs = n_selected * (n_selected - 1) // 2
    pair_i = np.empty(n_pairs, dtype=np.int32)
    pair_j = np.empty(n_pairs, dtype=np.int32)
    directions = np.empty(n_pairs, dtype=np.int8)
    delta_num = np.empty(n_pairs, dtype=np.int64)
    gamma = np.empty(n_pairs, dtype=np.float64)
    p_lt_negative = np.empty(n_pairs, dtype=np.float64)
    p_lt_positive = np.empty(n_pairs, dtype=np.float64)
    for local_i in prange(n_selected - 1):
        feature_i = features[local_i]
        base = local_i * (n_selected - 1) - (local_i * (local_i - 1)) // 2

        for local_j in range(local_i + 1, n_selected):
            feature_j = features[local_j]
            pair_idx = base + (local_j - local_i - 1)

            lt_negative = 0
            lt_positive = 0
            rank_diff_negative = 0.0
            rank_diff_positive = 0.0

            for sample_idx in range(x.shape[0]):
                xi_lt_xj = x[sample_idx, feature_i] < x[sample_idx, feature_j]
                rank_diff = ranks[sample_idx, feature_i] - ranks[sample_idx, feature_j]
                if y01[sample_idx] == 0:
                    lt_negative += xi_lt_xj
                    rank_diff_negative += rank_diff
                else:
                    lt_positive += xi_lt_xj
                    rank_diff_positive += rank_diff

            signed_delta_num = lt_positive * n_negative - lt_negative * n_positive
            if signed_delta_num < 0:
                delta_num[pair_idx] = -signed_delta_num
                directions[pair_idx] = np.int8(-1)
            else:
                delta_num[pair_idx] = signed_delta_num
                directions[pair_idx] = np.int8(1)

            mean_diff_negative = rank_diff_negative / n_negative
            mean_diff_positive = rank_diff_positive / n_positive
            gamma[pair_idx] = abs(mean_diff_positive - mean_diff_negative)
            p_lt_negative[pair_idx] = lt_negative / float(n_negative)
            p_lt_positive[pair_idx] = lt_positive / float(n_positive)
            pair_i[pair_idx] = feature_i
            pair_j[pair_idx] = feature_j

    return pair_i, pair_j, directions, delta_num, gamma, p_lt_negative, p_lt_positive
