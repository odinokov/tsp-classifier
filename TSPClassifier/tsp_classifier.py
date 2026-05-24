from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.model_selection import LeaveOneOut, check_cv
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from ._kernels import _rank_rows_average_numba, _score_pairs_numba
from ._model import (
    _BinaryTSPModel,
    _positive_fraction,
    _predict_binary_model,
    _vote_matrix,
)


class TSPClassifier(ClassifierMixin, TransformerMixin, BaseEstimator):
    """Fast Top-Scoring Pairs classifier with a scikit-learn compatible API.

    Input matrices follow scikit-learn orientation: ``(n_samples, n_features)``.
    By default the estimator uses a rank-based feature screen before exact
    scoring of candidate pairs. Set ``exact_pairs=True`` to score all pairs.
    """

    def __init__(
        self,
        n_pairs: int | str = 1,
        *,
        max_pairs: int = 10,
        cv: Any | None = None,
        multiclass: str = "ovr",
        exact_pairs: bool = False,
        max_features: int | None = 512,
    ) -> None:
        self.n_pairs = n_pairs
        self.max_pairs = max_pairs
        self.cv = cv
        self.multiclass = multiclass
        self.exact_pairs = exact_pairs
        self.max_features = max_features

    def fit(self, X: np.ndarray, y: np.ndarray) -> TSPClassifier:
        X_checked, y_checked = check_X_y(X, y, dtype=np.float64, ensure_all_finite=True)
        check_classification_targets(y_checked)
        X_checked = np.ascontiguousarray(X_checked, dtype=np.float64)
        self._validate_parameters()

        self.classes_ = np.unique(y_checked)
        self.n_features_in_ = X_checked.shape[1]
        if self.classes_.size < 2:
            raise ValueError("TSPClassifier needs at least two classes.")

        if self.classes_.size == 2:
            y01 = (y_checked == self.classes_[1]).astype(np.int32)
            model = self._fit_binary(X_checked, y01, self.classes_[0], self.classes_[1])
            self.estimators_ = [model]
            self.tasks_ = [(self.classes_[0], self.classes_[1])]
            self.pairs_ = model.pairs
            self.directions_ = model.directions
            self.delta_ = model.delta
            self.gamma_ = model.gamma
            self.k_ = model.k
            self.candidate_features_ = model.candidate_features
            return self

        if self.multiclass == "ovr":
            self.estimators_ = []
            self.tasks_ = []
            for cls in self.classes_:
                y01 = (y_checked == cls).astype(np.int32)
                model = self._fit_binary(X_checked, y01, None, cls)
                self.estimators_.append(model)
                self.tasks_.append((None, cls))
        elif self.multiclass == "ovo":
            self.estimators_ = []
            self.tasks_ = []
            for negative_class, positive_class in combinations(self.classes_, 2):
                mask = (y_checked == negative_class) | (y_checked == positive_class)
                y_pair = y_checked[mask]
                y01 = (y_pair == positive_class).astype(np.int32)
                model = self._fit_binary(
                    np.ascontiguousarray(X_checked[mask], dtype=np.float64),
                    y01,
                    negative_class,
                    positive_class,
                )
                self.estimators_.append(model)
                self.tasks_.append((negative_class, positive_class))
        else:
            raise ValueError("multiclass must be 'ovr' or 'ovo'.")

        self.k_ = np.array([model.k for model in self.estimators_], dtype=np.int32)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "estimators_")
        X_checked = self._check_predict_input(X)

        if self.classes_.size == 2:
            return _predict_binary_model(X_checked, self.estimators_[0])

        if self.multiclass == "ovr":
            scores = np.column_stack(
                [_positive_fraction(X_checked, model) for model in self.estimators_]
            )
            return self.classes_[np.argmax(scores, axis=1)]

        votes = np.zeros((X_checked.shape[0], self.classes_.size), dtype=np.int32)
        margins = np.zeros((X_checked.shape[0], self.classes_.size), dtype=np.float64)
        class_to_index = {cls: idx for idx, cls in enumerate(self.classes_)}

        for model in self.estimators_:
            negative_idx = class_to_index[model.negative_class]
            positive_idx = class_to_index[model.positive_class]
            score = _positive_fraction(X_checked, model)
            positive_vote = score > 0.5
            votes[positive_vote, positive_idx] += 1
            votes[~positive_vote, negative_idx] += 1
            margin = score - 0.5
            margins[:, positive_idx] += margin
            margins[:, negative_idx] -= margin

        pred_idx = np.empty(X_checked.shape[0], dtype=np.intp)
        for sample_idx in range(X_checked.shape[0]):
            best_idx = 0
            best_votes = votes[sample_idx, 0]
            best_margin = margins[sample_idx, 0]
            for class_idx in range(1, self.classes_.size):
                class_votes = votes[sample_idx, class_idx]
                class_margin = margins[sample_idx, class_idx]
                if class_votes > best_votes or (
                    class_votes == best_votes and class_margin > best_margin
                ):
                    best_idx = class_idx
                    best_votes = class_votes
                    best_margin = class_margin
            pred_idx[sample_idx] = best_idx
        return self.classes_[pred_idx]

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "estimators_")
        X_checked = self._check_predict_input(X)
        blocks = [_vote_matrix(X_checked, model) for model in self.estimators_]
        return np.hstack(blocks).astype(np.int8, copy=False)

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "estimators_")
        X_checked = self._check_predict_input(X)
        if self.classes_.size == 2:
            return _positive_fraction(X_checked, self.estimators_[0]) - 0.5
        if self.multiclass == "ovr":
            return np.column_stack(
                [_positive_fraction(X_checked, model) for model in self.estimators_]
            )

        votes = np.zeros((X_checked.shape[0], self.classes_.size), dtype=np.float64)
        class_to_index = {cls: idx for idx, cls in enumerate(self.classes_)}
        for model in self.estimators_:
            negative_idx = class_to_index[model.negative_class]
            positive_idx = class_to_index[model.positive_class]
            score = _positive_fraction(X_checked, model) - 0.5
            votes[:, positive_idx] += score
            votes[:, negative_idx] -= score
        return votes

    def _validate_parameters(self) -> None:
        if self.n_pairs != "auto":
            if not isinstance(self.n_pairs, int):
                raise TypeError("n_pairs must be a positive odd integer or 'auto'.")
            if self.n_pairs < 1 or self.n_pairs % 2 == 0:
                raise ValueError("n_pairs must be a positive odd integer.")

        if not isinstance(self.max_pairs, int) or self.max_pairs < 1:
            raise ValueError("max_pairs must be a positive integer.")
        if self.multiclass not in {"ovr", "ovo"}:
            raise ValueError("multiclass must be 'ovr' or 'ovo'.")
        if self.max_features is not None and self.max_features < 2:
            raise ValueError("max_features must be at least 2, or None.")

    def _check_predict_input(self, X: np.ndarray) -> np.ndarray:
        X_checked = check_array(X, dtype=np.float64, ensure_all_finite=True)
        if X_checked.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_checked.shape[1]} features; expected {self.n_features_in_}."
            )
        return np.ascontiguousarray(X_checked, dtype=np.float64)

    def _fit_binary(
        self,
        X: np.ndarray,
        y01: np.ndarray,
        negative_class: Any,
        positive_class: Any,
    ) -> _BinaryTSPModel:
        counts = np.bincount(y01, minlength=2)
        if counts[0] == 0 or counts[1] == 0:
            raise ValueError("Each binary TSP task must contain both classes.")

        max_possible = X.shape[1] // 2
        if max_possible < 1:
            raise ValueError("TSPClassifier needs at least two features.")

        if self.n_pairs == "auto":
            max_k = min(self.max_pairs, max_possible)
            if max_k % 2 == 0:
                max_k -= 1
            if max_k < 1:
                raise ValueError("No odd k is available for this feature count.")
            if counts.min() < 2:
                raise ValueError("n_pairs='auto' needs at least two samples per binary class.")
            k = self._choose_k_by_cv(X, y01, max_k)
        else:
            k = int(self.n_pairs)
            if k > max_possible:
                raise ValueError(f"n_pairs={k} needs at least {2 * k} features; got {X.shape[1]}.")

        return self._fit_binary_fixed(X, y01, negative_class, positive_class, k)

    def _choose_k_by_cv(self, X: np.ndarray, y01: np.ndarray, max_k: int) -> int:
        k_values = np.arange(1, max_k + 1, 2, dtype=np.int32)
        errors = np.zeros(k_values.size, dtype=np.int64)
        total = 0

        splitter = LeaveOneOut() if self.cv is None else check_cv(self.cv, y01, classifier=True)
        for train_idx, test_idx in splitter.split(X, y01):
            y_train = y01[train_idx]
            counts = np.bincount(y_train, minlength=2)
            if counts[0] == 0 or counts[1] == 0:
                raise ValueError("A CV split produced a training fold with one class.")

            fold_model = self._fit_binary_fixed(X[train_idx], y_train, 0, 1, max_k)
            for k_idx, k in enumerate(k_values):
                score = _positive_fraction(X[test_idx], fold_model, upto=int(k))
                pred = (score > 0.5).astype(np.int32)
                errors[k_idx] += int(np.sum(pred != y01[test_idx]))
            total += test_idx.size

        if total == 0:
            raise ValueError("CV splitter produced no test samples.")
        return int(k_values[np.argmin(errors)])

    def _fit_binary_fixed(
        self,
        X: np.ndarray,
        y01: np.ndarray,
        negative_class: Any,
        positive_class: Any,
        k: int,
    ) -> _BinaryTSPModel:
        ranks = _rank_rows_average_numba(X)
        features = self._candidate_features(ranks, y01, k)
        if features.size < 2 * k:
            raise ValueError(
                f"Need at least {2 * k} candidate features to select {k} disjoint pairs."
            )

        counts = np.bincount(y01, minlength=2)
        pair_i, pair_j, directions, delta_num, gamma, p0, p1 = _score_pairs_numba(
            X,
            ranks,
            y01.astype(np.int32, copy=False),
            features.astype(np.int32, copy=False),
            int(counts[0]),
            int(counts[1]),
        )

        order = np.lexsort((pair_j, pair_i, -gamma, -delta_num))
        used = np.zeros(X.shape[1], dtype=bool)
        selected: list[int] = []
        for pair_idx in order:
            gi = int(pair_i[pair_idx])
            gj = int(pair_j[pair_idx])
            if used[gi] or used[gj]:
                continue
            selected.append(int(pair_idx))
            used[gi] = True
            used[gj] = True
            if len(selected) == k:
                break

        if len(selected) != k:
            raise ValueError(f"Could only select {len(selected)} disjoint pairs, expected {k}.")

        selected_idx = np.asarray(selected, dtype=np.intp)
        denom = float(counts[0] * counts[1])
        return _BinaryTSPModel(
            negative_class=negative_class,
            positive_class=positive_class,
            pairs=np.column_stack((pair_i[selected_idx], pair_j[selected_idx])).astype(np.int32),
            directions=directions[selected_idx].astype(np.int8),
            delta=delta_num[selected_idx].astype(np.float64) / denom,
            gamma=gamma[selected_idx].astype(np.float64),
            p_lt_negative=p0[selected_idx].astype(np.float64),
            p_lt_positive=p1[selected_idx].astype(np.float64),
            candidate_features=features.astype(np.int32, copy=False),
            k=k,
        )

    def _candidate_features(self, ranks: np.ndarray, y01: np.ndarray, k: int) -> np.ndarray:
        n_features = ranks.shape[1]
        if self.exact_pairs or self.max_features is None or self.max_features >= n_features:
            return np.arange(n_features, dtype=np.int32)

        candidate_count = min(n_features, max(int(self.max_features), 2 * k))
        negative = y01 == 0
        positive = y01 == 1
        rank_shift = np.abs(ranks[positive].mean(axis=0) - ranks[negative].mean(axis=0))
        order = np.lexsort((np.arange(n_features), -rank_shift))
        return np.sort(order[:candidate_count]).astype(np.int32)
