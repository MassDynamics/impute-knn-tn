# CLAUDE.md

## Overview

Python reimplementation of the **kNN-TN (k-Nearest Neighbours with Truncated Normal) imputation** algorithm from the [GSimp package](https://github.com/WandeRum/GSimp) (Wei et al., 2018). Canonical R source: `Trunc_KNN/Imput_funcs.r`.

## Architecture

```
src/impute_knn_tn/
  impute.py           # Public API: impute_knn_tn(), knn_tn()
  knn_engine.py       # Core kNN loop: impute_knn()
  truncnorm_mle.py    # mklhood + Newton-Raphson + EstimatesComputation
```

## Commands

```sh
# Install (editable, with dev deps)
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Dev CLI tools (installed globally via uv, not in the venv)
uv tool install ruff
uv tool install ty
# uv itself: https://docs.astral.sh/uv/getting-started/installation/

# Install pre-commit hooks (ruff lint/format, ty type check, nbstripout)
pre-commit install

# Run tests (excluding slow synthetic_5000/20000 tests)
pytest tests/ -v -m "not slow"

# Run all tests including slow ones
pytest tests/ -v

# Lint / format / type check
ruff check src/ tests/
ruff format src/ tests/
ty check src/

# Run benchmarks
python dev/benchmark.py
```

## Key conventions

- All parity tests compare against R reference CSVs in `tests/reference/`
- R references were generated from MDImputeKnnTn repo (`dev/generate_r_reference.R`) with `mklhood` added from GSimp
- GSimp datasets (targeted, untargeted, real_data, data_sim) are the primary parity tests — they exercise the Newton-Raphson/mklhood path
- Integration tolerances match R's `integrate()` defaults (`eps^0.25 ≈ 1.22e-4`) for numerical parity
- `np.where` returns row-first order; R's `which(arr.ind=TRUE)` returns column-first — the kNN loop sorts by column to match R

## Status

- 38 non-slow tests + 8 slow tests (synthetic_5000/20000 parity) all pass
- Parity tests at `rtol=1e-10` for matrix-level, `rtol=1e-6` for param estimates
- Performance: comparable to R for ≤1K features, ~5x slower at 5K+ (Python loop overhead)
- Next: Numba/Cython optimization of correlation loop, README, LICENSE
