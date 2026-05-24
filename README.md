# TSPClassifier

`TSPClassifier` is a scikit-learn compatible implementation of Top-Scoring Pairs
(TSP/k-TSP) classifiers for gene expression and other high-dimensional
biological matrices.

## When To Use TSP

Top-Scoring Pairs is a rule-based classifier for high-dimensional data where
the relative order of features within each sample is informative. Instead of
learning weights on absolute feature values, it searches for feature pairs
`(i, j)` whose ordering, such as `x_i < x_j`, differs strongly between classes.
A k-TSP model selects an odd number of high-scoring, non-overlapping pairs and
predicts by majority vote across those pair rules.

This makes TSP useful for gene expression, proteomics, metabolomics, NDR scores,
and other same-scale biological matrices where simple, interpretable, robust
within-sample comparisons are preferred over black-box models. It is especially
appropriate when the number of features is large, the number of samples is
modest, and absolute values may be affected by normalization or batch effects.

TSP is not suitable when features are not meaningfully comparable within a
sample, when predictive signal depends mainly on absolute magnitude rather than
relative ordering, or when the goal is calibrated probabilities rather than
transparent decision rules.

The main estimator is `TSPClassifier`.

```text
X.shape == (n_samples, n_features)
```

Rows are samples or cells. Columns are features whose values are comparable
within each sample, such as genes from one expression matrix, proteins or
metabolites from one normalized panel, NDR scores, or other same-scale
assay-derived features.

## Install

From this checkout:

```bash
pip install .
```

From GitHub:

```bash
pip install git+https://github.com/odinokov/TSPClassifier.git
```

For development:

```bash
poetry install
```

Core dependencies:

```bash
pip install numpy numba scipy scikit-learn
```

Optional packages used in examples:

```bash
pip install decoupler anndata pandas skops joblib
```

## Quick Start

```python
import numpy as np
from TSPClassifier import TSPClassifier

X = np.array(
    [
        [2.0, 3.0, 1.0, 4.0],
        [2.0, 3.0, 1.0, 4.0],
        [3.0, 2.0, 4.0, 1.0],
        [3.0, 2.0, 4.0, 1.0],
    ]
)
y = np.array([0, 0, 1, 1])

clf = TSPClassifier(n_pairs=1, exact_pairs=True)
clf.fit(X, y)

print(clf.predict(X))
print(clf.pairs_)
print(clf.delta_)
print(clf.gamma_)
```

Expected output:

```text
[0 0 1 1]
[[2 3]]
[1.]
[6.]
```

## MiceProtein OpenML Example

`MiceProtein` is a useful real-data example for TSP because it contains 77
same-scale protein or protein-modification measurements from mouse cerebral
cortex. The ground truth is the experimental class label, combining genotype,
learning stimulation, and treatment.

```python
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.impute import KNNImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline

from TSPClassifier import TSPClassifier

mice = fetch_openml(name="miceprotein", version=4, as_frame=True, parser="auto")
X = mice.data.to_numpy(dtype=float)
y = mice.target.to_numpy()

clf = make_pipeline(
    KNNImputer(n_neighbors=5),
    TSPClassifier(
        n_pairs="auto",
        max_pairs=9,
        cv=5,
        exact_pairs=True,
        multiclass="ovo",
    ),
)

outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print(cross_val_score(clf, X, y, cv=outer_cv).mean())
```

For this example, `max_pairs=9` was selected by sweeping odd ceilings and
keeping the highest mean 5-fold accuracy.

```python
sweep_scores = {}

for max_pairs in range(1, 16, 2):
    sweep_clf = make_pipeline(
        KNNImputer(n_neighbors=5),
        TSPClassifier(
            n_pairs="auto",
            max_pairs=max_pairs,
            cv=5,
            exact_pairs=True,
            multiclass="ovo",
        ),
    )
    score = cross_val_score(sweep_clf, X, y, cv=outer_cv).mean()
    sweep_scores[max_pairs] = score

best_max_pairs = max(sweep_scores, key=sweep_scores.get)
print(best_max_pairs, f"{sweep_scores[best_max_pairs]:.4f}")
```

On one fixed 5-fold split, this gave:

| Model | Mean accuracy |
| --- | ---: |
| `TSPClassifier`, one-vs-one | 0.7935 |
| Logistic regression | 0.9898 |
| RBF SVC | 0.9972 |
| Random forest | 0.9944 |
| Extra trees | 0.9991 |

On this dataset, flexible baselines classify the experimental groups almost
perfectly. TSP is lower-accuracy but returns transparent protein-pair ordering
rules instead of opaque multifeature decision boundaries.

## API

```python
TSPClassifier(
    n_pairs=1,
    *,
    max_pairs=10,
    cv=None,
    multiclass="ovr",
    exact_pairs=False,
    max_features=512,
)
```

Parameters:

- `n_pairs`: positive odd integer, or `"auto"`. `1` is the single-pair TSP
  classifier. `"auto"` selects k by cross-validation.
- `max_pairs`: maximum k considered when `n_pairs="auto"`. Candidate k values
  are odd values up to this limit.
- `cv`: optional scikit-learn CV splitter or integer fold count for
  `n_pairs="auto"`. Default `None` uses leave-one-out cross-validation.
- `multiclass`: `"ovr"` for one-vs-rest or `"ovo"` for one-vs-one multiclass
  decomposition.
- `exact_pairs`: scoring-mode switch.
  - `True`: score every possible pair among all input features. This is the
    direct Exact All-Pairs mode, and `max_features` is ignored.
  - `False`: use candidate-screened mode. Features are screened first, then
    every pair among the retained candidate features is scored exactly.
- `max_features`: candidate feature limit used only when `exact_pairs=False`.
  Use an integer such as `512` or `1000` for fast screening. Use `None` to
  disable the screening step inside candidate-screened mode, so all features are
  retained as candidates.

Learned attributes for binary models:

- `pairs_`: selected feature-index pairs, shape `(k, 2)`.
- `directions_`: selected comparison direction per pair.
- `delta_`: primary TSP score for each selected pair.
- `gamma_`: rank-difference tie-break score for each selected pair.
- `k_`: selected number of pairs.
- `candidate_features_`: features retained before pair scoring in screened
  mode.

For multiclass models, binary submodels are stored in `estimators_`, and their
tasks are stored in `tasks_`.

## Scoring Modes

There are three practical configurations:

| Configuration | What happens | Typical use |
| --- | --- | --- |
| `exact_pairs=True` | Directly scores all `p * (p - 1) / 2` feature pairs. `max_features` is ignored. | Exact reference runs, moderate feature counts. |
| `exact_pairs=False, max_features=N` | Keeps up to `N` candidate features, then scores all pairs among those candidates. | Default fast mode for high-dimensional matrices. |
| `exact_pairs=False, max_features=None` | Keeps all features as candidates, then scores all pairs through the candidate-screened code path. | Small datasets where you want the screened-mode API but no prefilter. |

### Exact All-Pairs Scoring

Use this when the feature space is moderate or when you need a fully exact
all-pairs scan. This is the clearest way to request exact TSP pair scoring.

```python
clf = TSPClassifier(n_pairs=3, exact_pairs=True)
clf.fit(X_train, y_train)

print(clf.pairs_)
print(clf.delta_)
print(clf.gamma_)
```

`max_features` has no effect in this mode:

```python
clf = TSPClassifier(n_pairs=3, exact_pairs=True, max_features=512)
# max_features is ignored because exact_pairs=True.
```

### Slow Candidate-Screened Scoring For Small Feature Spaces

For small feature-space datasets, this mode keeps the candidate-screened branch
but disables the screening step by setting `max_features=None`. Every feature is
retained as a candidate, then all candidate pairs are scored.

This is not the recommended way to ask for exact all-pairs scoring. Prefer
`exact_pairs=True` for that. Use this mode only when you specifically want to
exercise the same `exact_pairs=False` API branch you plan to use later with
`max_features=N`.

```python
clf = TSPClassifier(
    n_pairs=3,
    exact_pairs=False,
    max_features=None,
)
clf.fit(X_train, y_train)

print(clf.candidate_features_)
print(clf.pairs_)
print(clf.delta_)
print(clf.gamma_)
```

Because all features are retained, this can still be slow when `p` is large.

### Fast Candidate-Screened Scoring

This is the default mode for larger expression matrices. The model first keeps
up to `max_features` candidates, then scores every pair among those retained
features.

```python
clf = TSPClassifier(
    n_pairs=3,
    exact_pairs=False,
    max_features=1000,
)
clf.fit(X_train, y_train)

print(clf.candidate_features_)
print(clf.pairs_)
```

## Choosing k

Let the model choose k by cross-validation:

```python
clf = TSPClassifier(n_pairs="auto", max_pairs=9, cv=5)
clf.fit(X_train, y_train)

print("selected k:", clf.k_)
print("selected pairs:", clf.pairs_)
```

Or use a scikit-learn splitter:

```python
from sklearn.model_selection import StratifiedKFold
from TSPClassifier import TSPClassifier

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

clf = TSPClassifier(
    n_pairs="auto",
    max_pairs=9,
    cv=cv,
    exact_pairs=True,
)
clf.fit(X_train, y_train)
```

## Multiclass Classification

One-vs-rest:

```python
clf = TSPClassifier(n_pairs=1, multiclass="ovr")
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
scores = clf.decision_function(X_test)
```

One-vs-one:

```python
clf = TSPClassifier(n_pairs=1, multiclass="ovo")
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
pair_vote_features = clf.transform(X_test)
```

For `C` classes, one-vs-one trains `C * (C - 1) / 2` binary TSP models.

## Use With scikit-learn

```python
from sklearn.model_selection import cross_val_score
from TSPClassifier import TSPClassifier

clf = TSPClassifier(n_pairs=1, exact_pairs=True)
scores = cross_val_score(clf, X, y, cv=5)

print(scores.mean())
```

`transform(X)` returns pair-vote features:

```python
clf.fit(X_train, y_train)
pair_votes = clf.transform(X_test)
print(pair_votes.shape)
```

## Using Scores To Choose Top Features

TSP is pairwise. The primary output is a ranked set of feature pairs rather than
a univariate feature list.

Use scores this way:

- `delta_`: primary discriminative score. Larger is better.
- `gamma_`: secondary score used to break ties among pairs with equal `delta_`.
- `pairs_`: feature indices selected for the fitted classifier.
- `n_pairs="auto"`: preferred way to choose model size by validation.

### Inspect Pair-Level Scores

```python
clf = TSPClassifier(n_pairs=5, exact_pairs=True).fit(X_train, y_train)

for rank, ((i, j), delta, gamma) in enumerate(
    zip(clf.pairs_, clf.delta_, clf.gamma_),
    start=1,
):
    print(
        {
            "rank": rank,
            "feature_i": int(i),
            "feature_j": int(j),
            "delta": float(delta),
            "gamma": float(gamma),
        }
    )
```

### Get The Selected Feature Set

```python
import numpy as np

top_features = np.unique(clf.pairs_.ravel())
print(top_features)
```

With feature names:

```python
feature_names = np.asarray(expression_matrix.columns)
print(feature_names[top_features])
```

### Rank Selected Features By Their Best Pair Score

This converts selected pairs into a feature-level summary by assigning each
feature its best pair score.

```python
import numpy as np

def rank_selected_features(clf, n_features):
    best_delta = np.full(n_features, -np.inf)
    best_gamma = np.full(n_features, -np.inf)
    support = np.zeros(n_features, dtype=int)

    for (i, j), delta, gamma in zip(clf.pairs_, clf.delta_, clf.gamma_):
        i = int(i)
        j = int(j)
        delta = float(delta)
        gamma = float(gamma)

        for feature_idx in (i, j):
            support[feature_idx] += 1
            if delta > best_delta[feature_idx] or (
                delta == best_delta[feature_idx]
                and gamma > best_gamma[feature_idx]
            ):
                best_delta[feature_idx] = delta
                best_gamma[feature_idx] = gamma

    selected = np.flatnonzero(support)
    order = np.lexsort(
        (
            selected,
            -best_gamma[selected],
            -best_delta[selected],
            -support[selected],
        )
    )
    return selected[order], best_delta[selected][order], best_gamma[selected][order]


features, deltas, gammas = rank_selected_features(clf, X_train.shape[1])

for feature_idx, delta, gamma in zip(features, deltas, gammas):
    print(f"feature_{feature_idx}: best Delta={delta:.3f}, best Gamma={gamma:.3f}")
```

Interpret this as a reporting summary. The pair rules remain the actual model.

### Rank Features Across Multiclass Models

```python
import numpy as np

def rank_multiclass_features(clf, n_features):
    best_delta = np.full(n_features, -np.inf)
    best_gamma = np.full(n_features, -np.inf)
    support = np.zeros(n_features, dtype=int)

    for model in clf.estimators_:
        for (i, j), delta, gamma in zip(model.pairs, model.delta, model.gamma):
            i = int(i)
            j = int(j)
            delta = float(delta)
            gamma = float(gamma)

            for feature_idx in (i, j):
                support[feature_idx] += 1
                if delta > best_delta[feature_idx] or (
                    delta == best_delta[feature_idx]
                    and gamma > best_gamma[feature_idx]
                ):
                    best_delta[feature_idx] = delta
                    best_gamma[feature_idx] = gamma

    selected = np.flatnonzero(support)
    order = np.lexsort(
        (
            selected,
            -best_gamma[selected],
            -best_delta[selected],
            -support[selected],
        )
    )
    return selected[order]


# X_multi and y_multi are a multiclass feature matrix and label vector.
clf = TSPClassifier(n_pairs=3, multiclass="ovo").fit(X_multi, y_multi)
top_features = rank_multiclass_features(clf, X_multi.shape[1])
print(top_features[:20])
```

For multiclass models, `support` is useful because the same feature can appear
in several one-vs-one or one-vs-rest binary tasks.

## Save And Load Models

Always save the fitted model together with the feature names and package
versions used to train it. `TSPClassifier` stores feature indices, so preserving
the column order is essential.

### Recommended: `skops` For Safer Persistence

`skops` is designed for safer scikit-learn model persistence because it avoids
loading arbitrary pickle payloads by default.

```python
# pip install skops

import json
import platform
from pathlib import Path

import numpy as np
import skops.io as sio
import sklearn
import TSPClassifier as tspclassifier
from TSPClassifier import TSPClassifier

model_path = "tsp_model.skops"
metadata_path = "tsp_model_metadata.json"

clf = TSPClassifier(n_pairs="auto", max_pairs=9, cv=5).fit(X_train, y_train)
sio.dump(clf, model_path)

metadata = {
    "feature_names": list(map(str, feature_names)),
    "classes": list(map(str, clf.classes_)),
    "python": platform.python_version(),
    "numpy": np.__version__,
    "scikit_learn": sklearn.__version__,
    "TSPClassifier": tspclassifier.__version__,
}
Path(metadata_path).write_text(json.dumps(metadata, indent=2))
```

Load only after reviewing trusted types:

```python
import json
from pathlib import Path

import skops.io as sio

unknown_types = sio.get_untrusted_types(file="tsp_model.skops")
# Review unknown_types before trusting them.
clf = sio.load("tsp_model.skops", trusted=unknown_types)

metadata = json.loads(Path("tsp_model_metadata.json").read_text())
expected_features = metadata["feature_names"]

X_new = expression_matrix.loc[:, expected_features].to_numpy(dtype=float)
y_pred = clf.predict(X_new)
```

### Trusted Internal Use: `joblib`

Use `joblib` only for files you trust. It is pickle-based and can execute
arbitrary code while loading malicious files.

```python
from joblib import dump, load

dump(clf, "tsp_model.joblib", compress=3)

# Only load from a trusted source.
clf = load("tsp_model.joblib")
```

Recommended persistence checklist:

- Save model and metadata together.
- Save feature names and enforce the same column order at inference.
- Pin package versions in `requirements.txt`, `pyproject.toml`, or an
  environment file.
- Do not load pickle/joblib files from untrusted sources.
- Prefer `skops` when model files cross trust boundaries.

## Algorithm Notes

For each feature pair `(i, j)`, TSP scores the strict within-sample ordering
event:

```text
x_i < x_j
```

The primary score is:

```text
Delta_ij = |P(x_i < x_j | class 1) - P(x_i < x_j | class 0)|
```

When pairs tie on `Delta`, `TSPClassifier` uses the rank-difference score:

```text
Gamma_ij = |mean( rank(x_i) - rank(x_j) | class 1 ) - mean( rank(x_i) - rank(x_j) | class 0 )|
```

where ranks are within-sample average ranks (ties broken by averaging).

Prediction uses majority vote over selected pair rules. k is odd to avoid vote
ties.

The expensive ranking and pair-scoring kernels are compiled with Numba and run
in parallel. That makes the classifier multi-threaded, but actual core usage
depends on Numba settings, workload size, and the available CPU resources; it
does not guarantee that every core will be saturated in every run.

## Validation

Check package metadata:

```bash
poetry check
```

Run a direct import and prediction smoke test:

```bash
python3 -c "import numpy as np; from TSPClassifier import TSPClassifier; X=np.array([[2.,3.,1.,4.],[2.,3.,1.,4.],[3.,2.,4.,1.],[3.,2.,4.,1.]]); y=np.array([0,0,1,1]); clf=TSPClassifier(n_pairs=1, exact_pairs=True).fit(X,y); print(clf.pairs_); print(clf.predict(X))"
```

## References

- Shi, P., Ray, S., Zhu, Q., & Kon, M. A. (2011). [Top scoring pairs for feature selection in machine learning and applications to cancer outcome prediction](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-12-375). *BMC Bioinformatics*, 12, 375. https://doi.org/10.1186/1471-2105-12-375
- Tan, A. C., Naiman, D. Q., Xu, L., Winslow, R. L., & Geman, D. (2005). [Simple decision rules for classifying human cancers from gene expression profiles](https://academic.oup.com/bioinformatics/article/21/20/3896/203010). *Bioinformatics*, 21(20), 3896-3904. https://doi.org/10.1093/bioinformatics/bti631
- Gui, T. (2014). [A Pairwise Feature Selection Method For Gene Data Using Information Gain](https://egrove.olemiss.edu/etd/943/). M.S. thesis, University of Mississippi, Electronic Theses and Dissertations 943.
- Bari, M. G., Salekin, S., & Zhang, J. (2017). [A Robust and Efficient Feature Selection Algorithm for Microarray Data](https://onlinelibrary.wiley.com/doi/10.1002/minf.201600099). *Molecular Informatics*, 36, 1600099. https://doi.org/10.1002/minf.201600099
- Lin, X., Huang, X., Zhou, L., Ren, W., Zeng, J., Yao, W., & Wang, X. (2019). [The Robust Classification Model Based on Combinatorial Features](https://ieeexplore.ieee.org/document/8126830/). *IEEE/ACM Transactions on Computational Biology and Bioinformatics*, 16(2), 650-657. https://doi.org/10.1109/TCBB.2017.2779512

## License

MIT License.
