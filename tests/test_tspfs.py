"""Regression + functional tests for TSPClassifier.

Golden values were captured from the pre-simplification implementation; any
behavioral drift in predict / decision_function / transform is a regression.
"""

from __future__ import annotations

import numpy as np
import pytest

from tspfs import TSPClassifier

_OVR_PRED = [0, 0, 0, 1, 2, 1, 0, 3, 1, 2, 2, 1, 3, 3, 2, 0, 2, 1, 2, 0, 1, 0, 0,
             2, 0, 2, 0, 0, 0, 0, 0, 2, 0, 3, 0, 1, 2, 0, 1, 3, 0, 2, 0, 1, 3, 0,
             0, 2, 1, 0, 0, 3, 2, 1, 1, 0, 0, 0, 2, 0]
_OVO_PRED = [3, 3, 0, 1, 0, 1, 2, 3, 1, 0, 0, 1, 3, 3, 2, 0, 2, 1, 2, 0, 1, 1, 0,
             2, 2, 2, 3, 3, 1, 1, 0, 2, 0, 3, 1, 1, 2, 0, 1, 2, 3, 3, 3, 3, 0, 1,
             1, 2, 1, 0, 0, 3, 2, 1, 1, 0, 1, 1, 2, 0]
_BIN_PRED = [0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1,
             1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0]


@pytest.fixture(scope="module")
def multiclass_data():
    rng = np.random.RandomState(0)
    X = rng.rand(60, 12)
    y = rng.randint(0, 4, size=60)
    # Reproduce the exact RNG draw order used to capture the golden values.
    Xb = rng.rand(40, 8)
    yb = rng.randint(0, 2, size=40)
    return X, y, Xb, yb


@pytest.mark.parametrize("multiclass,golden", [("ovr", _OVR_PRED), ("ovo", _OVO_PRED)])
def test_multiclass_predict_regression(multiclass_data, multiclass, golden):
    X, y, _, _ = multiclass_data
    clf = TSPClassifier(n_pairs=3, multiclass=multiclass).fit(X, y)
    assert clf.predict(X).tolist() == golden


def test_binary_predict_regression(multiclass_data):
    _, _, Xb, yb = multiclass_data
    clf = TSPClassifier(n_pairs=3).fit(Xb, yb)
    assert clf.predict(Xb).tolist() == _BIN_PRED
    assert round(float(clf.decision_function(Xb).sum()), 6) == 1.666667


@pytest.mark.parametrize(
    "multiclass,dec_sum,trans_sum,trans_cols",
    [("ovr", 116.666667, 350, 12), ("ovo", -0.0, 529, 18)],
)
def test_multiclass_shapes_regression(multiclass_data, multiclass, dec_sum, trans_sum, trans_cols):
    X, y, _, _ = multiclass_data
    clf = TSPClassifier(n_pairs=3, multiclass=multiclass).fit(X, y)
    dec = clf.decision_function(X)
    trans = clf.transform(X)
    assert dec.shape == (60, 4)
    assert round(float(dec.sum()), 6) == dec_sum
    assert trans.shape == (60, trans_cols)
    assert int(trans.sum()) == trans_sum
    assert trans.dtype == np.int8


def test_fit_transform_matches_fit_then_transform(multiclass_data):
    X, y, _, _ = multiclass_data
    a = TSPClassifier(n_pairs=3, multiclass="ovo").fit(X, y).transform(X)
    b = TSPClassifier(n_pairs=3, multiclass="ovo").fit_transform(X, y)
    np.testing.assert_array_equal(a, b)


def test_n_pairs_auto_runs(multiclass_data):
    _, _, Xb, yb = multiclass_data
    clf = TSPClassifier(n_pairs="auto", max_pairs=3).fit(Xb, yb)
    assert clf.k_ in {1, 3}
    assert clf.predict(Xb).shape == (40,)


@pytest.mark.parametrize(
    "kwargs,exc",
    [
        ({"n_pairs": 2}, ValueError),
        ({"n_pairs": 1.5}, TypeError),
        ({"max_pairs": 0}, ValueError),
        ({"multiclass": "bogus"}, ValueError),
        ({"max_features": 1}, ValueError),
    ],
)
def test_parameter_validation(multiclass_data, kwargs, exc):
    X, y, _, _ = multiclass_data
    with pytest.raises(exc):
        TSPClassifier(**kwargs).fit(X, y)
