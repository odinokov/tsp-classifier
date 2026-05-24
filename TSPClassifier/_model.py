from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class _BinaryTSPModel:
    negative_class: Any
    positive_class: Any
    pairs: np.ndarray
    directions: np.ndarray
    delta: np.ndarray
    gamma: np.ndarray
    p_lt_negative: np.ndarray
    p_lt_positive: np.ndarray
    candidate_features: np.ndarray
    k: int


def _positive_fraction(
    X: np.ndarray,
    model: _BinaryTSPModel,
    upto: int | None = None,
) -> np.ndarray:
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
    score = _positive_fraction(X, model)
    pred_positive = score > 0.5
    return np.where(pred_positive, model.positive_class, model.negative_class)
