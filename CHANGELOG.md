# Changelog

All notable changes from productionizing the research code into the installable
`tspfs` package. Each entry says **what** changed and **why**, so the rationale
is reusable beyond this one repo.

The governing rule throughout was **faithfulness**: the science must compute the
exact same numbers. Every change below is either behavior-neutral or was proven
so by the golden-value regression suite (12/12 passing, byte-identical output).

## [0.1.0] — 2026-06-17 — Research → production package

### Packaging
- **Poetry → PEP 621 + hatchling** (`pyproject.toml`).
  *Why:* `[project]` is the standardized, tool-agnostic metadata table; any
  build frontend can read it. Poetry's `[tool.poetry]` locks you to one tool.
- **Flat `tsp_classifier/` → `src/tspfs/` (src layout).**
  *Why:* a `src/` layout makes `import tspfs` resolve to the *installed* package,
  not the working directory. Packaging mistakes (a module left out of the wheel)
  fail your own tests instead of surfacing only after a user `pip install`s.
- **Dropped the direct `scipy` dependency.**
  *Why:* it was declared but never imported — scikit-learn already pulls it in
  transitively. Declaring deps you don't use misleads readers and needlessly
  constrains the resolver. List only what you `import`.
- **Added `py.typed` (PEP 561) and a `[dev]` extra (pytest, mypy).**
  *Why:* without the `py.typed` marker, downstream type checkers silently ignore
  your annotations — the hints only count for consumers if you ship the marker.
- **Fixed `[project.urls].Repository` to the actual remote** (`odinokov/tsp-classifier`).
  *Why:* import name (`tspfs`), distribution name, and repo name can all differ;
  the metadata must point at where the code really lives, not an aspirational URL.

### Structure
- **Internal module `tsp_classifier.py` → `estimator.py`.**
  *Why:* `tspfs/tsp_classifier.py` was a redundant stutter. The public import
  `from tspfs import TSPClassifier` is unchanged because `__init__.py` re-exports —
  internal module names are free to be clear; the public surface is the contract.
- **New `errors.py`: `TSPError` / `TSPValueError(ValueError)` / `TSPTypeError(TypeError)`,**
  and all 17 explicit `raise` sites converted to them.
  *Why:* a typed hierarchy lets callers `except TSPError` for anything from this
  package. Subclassing the built-ins means existing `except ValueError` /
  `except TypeError` handlers keep working — you add capability with zero breakage.

### Typing
- **mypy `--strict`, with narrow, documented relaxations.**
  Turned off `disallow_any_generics` and `disallow_subclassing_any`, scoped
  `ignore_missing_imports` to `numba.*`/`sklearn.*`, and localized two
  `# type: ignore[untyped-decorator]` (on `@njit`) and two `cast()` calls (at the
  numpy-indexing seam).
  *Why:* numpy's `ndarray` generics, scikit-learn shipping no `py.typed`, and
  numba's untyped decorators make *literal* strict impossible without annotation
  noise that buys no safety. The right move is to relax the specific flags **with
  a comment explaining each**, and confine `ignore`/`cast` to the genuine untyped
  boundary — never a blanket `ignore_errors`.
- **Class-level annotations for the polymorphic fitted attrs** (`classes_`, `k_`).
  *Why:* `k_` is a scalar for binary tasks and an array for multiclass; declaring
  the union documents that intent and unblocks the checker without a runtime cost.

### Code quality (simplification pass)
- **Extracted `_ovo_votes_and_margins()`**, now shared by `predict` and
  `decision_function`.
  *Why:* both methods rebuilt the same `class_to_index` map and ran the identical
  one-vs-one accumulation loop. Duplicated logic drifts — one shared helper means
  one place to fix and no chance of the two paths disagreeing.
- **Removed dead `p_lt_negative` / `p_lt_positive`** from the Numba kernel, the
  `_BinaryTSPModel` dataclass, and the fit path (kernel return 7-tuple → 5-tuple).
  *Why:* they were computed, stored, and never read (verified by grep across
  `src/` and `tests/`). Dead state isn't free — here it cost two allocations and
  two divisions *inside the hot per-pair loop*, plus two `.astype` copies and
  noise in the kernel signature. Delete provably-dead code; keep-and-flag when
  unsure.
- **Kept the `score > 0.5` comparison instead of folding it to `margin > 0.0`.**
  *Why:* the two are algebraically equal but not guaranteed bit-identical in
  floating point. When faithfulness is the spec, preserve the exact operation —
  "obviously equivalent" math is where silent regressions hide.
- **Google-style docstrings on every public symbol.**
  *Why:* Sphinx-autodoc-compatible docstrings are documentation that can't rot out
  of sync with a separate doc site.

### Documentation
- README: import paths `tsp_classifier` → `tspfs`; `poetry install` →
  `pip install -e ".[dev]"`; `poetry check` → `python -m build`; dropped `scipy`
  from the manual-install line; added an **Errors** section documenting the
  exception hierarchy.
  *Why:* an undocumented public exception hierarchy is pure cost — you pay the
  API-stability burden of three more exported names without users knowing they can
  catch them. Document a contract or don't expose it.

### Testing / faithfulness
- Repointed the golden-value regression suite (`tests/test_tspfs.py`) to `tspfs`;
  **12/12 pass, output byte-identical** to the original.
  *Why:* the golden values captured from the pre-port implementation *are* the
  spec. Re-running them after the port, after the dead-code removal, and after the
  OvO refactor is what makes "I didn't change behavior" a verified fact rather than
  a hope. Re-verify after every behavior-touching edit, not just at the end.
- **Note on environment:** installing into the existing conda base upgraded
  numba 0.60 → 0.65 and numpy 2.0 → 2.2. This was an explicit choice to proceed;
  the regression suite confirms no output drift under the newer versions. The
  cleaner pattern is an isolated venv pinned to the original's resolved versions.

### Housekeeping
- Original sources and stale artifacts archived under `old/` (gitignored):
  the previous nested `tsp_classifier/` repo, a stale `tspfs/` memory dump, the
  old `AGENTS.md`, and a backup of the Poetry `pyproject.toml`. Nothing was
  destructively deleted.
