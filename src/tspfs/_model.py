"""Fitted binary-TSP model container and prediction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class _BinaryTSPModel:
    """A fitted set of ``k`` disjoint top-scoring pairs for one binary task.

    Attributes:
        negative_class: label voted for when a pair fires negative.
        positive_class: label voted for when a pair fires positive.
        pairs: ``(k, 2)`` int32 feature-index pairs, best first.
        directions: ``(k,)`` int8 in ``{-1, +1}``; +1 means ``x_i < x_j`` votes positive.
        delta: ``(k,)`` normalized score differences.
        gamma: ``(k,)`` rank-difference tie-breaker magnitudes.
        candidate_features: int32 feature indices that survived screening.
        k: number of selected pairs.
    """

    negative_class: Any
    positive_class: Any
    pairs: np.ndarray
    directions: np.ndarray
    delta: np.ndarray
    gamma: np.ndarray
    candidate_features: np.ndarray
    k: int


def _positive_fraction(
    X: np.ndarray,
    model: _BinaryTSPModel,
    upto: int | None = None,
) -> np.ndarray:
    """Fraction of the model's pairs that vote for the positive class.

    Args:
        X: ``(n_samples, n_features)`` float matrix.
        model: fitted binary model.
        upto: score only the first ``upto`` pairs; defaults to all ``k``.

    Returns:
        ``(n_samples,)`` float array in ``[0, 1]``.
    """
    k = model.k if upto is None else min(int(upto), model.k)
    votes = np.zeros(X.shape[0], dtype=np.float64)
    for pair_idx in range(k):
        gi = model.pairs[pair_idx, 0]
        gj = model.pairs[pair_idx, 1]
        if model.directions[pair_idx] == 1:
            votes += X[:, gi] < X[:, gj]
        else:
            votes += X[:, gi] > X[:, gj]
    return votes / float(k)


def _vote_matrix(X: np.ndarray, model: _BinaryTSPModel) -> np.ndarray:
    """Per-pair positive votes as an ``(n_samples, k)`` int8 matrix."""
    votes = np.empty((X.shape[0], model.k), dtype=np.int8)
    for pair_idx in range(model.k):
        gi = model.pairs[pair_idx, 0]
        gj = model.pairs[pair_idx, 1]
        if model.directions[pair_idx] == 1:
            votes[:, pair_idx] = X[:, gi] < X[:, gj]
        else:
            votes[:, pair_idx] = X[:, gi] > X[:, gj]
    return votes


def _predict_binary_model(X: np.ndarray, model: _BinaryTSPModel) -> np.ndarray:
    """Predict class labels for a single fitted binary model by majority vote."""
    score = _positive_fraction(X, model)
    pred_positive = score > 0.5
    return np.where(pred_positive, model.positive_class, model.negative_class)
