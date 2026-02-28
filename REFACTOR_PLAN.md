# ANUGA Tech-Debt Refactor Plan

**Branch:** `anuga-4.0-refactor-plan`
**Date:** 2026-02-28
**Fork:** [Hydrata/anuga_core](https://github.com/Hydrata/anuga_core)
**Upstream:** [anuga-community/anuga_core](https://github.com/anuga-community/anuga_core)

---

## Executive Summary

This plan modernises the ANUGA codebase in 5 phases (0–4), prioritising test safety, dependency hygiene, code deduplication, linting, and expanded test coverage. A dedicated section addresses fork/upstream synchronisation to ensure Hydrata's changes remain mergeable.

---

## Current State Assessment

### Dependencies
- **pyproject.toml declares only `numpy>=2.0.0`** — but the code actually imports scipy, netCDF4, meshpy, dill, pymetis, pyproj, affine, matplotlib, and more
- **Phantom dependencies:** cartopy and openpyxl appear in some code paths but are never actually imported
- **GDAL:** partially removed on the `remove-gdal` branch (replaced with rasterio/shapely), but remnants remain
- **requires-python:** `">3.8, <3.14"` is incorrect syntax and should be `">=3.9, <3.14"`

### Code Duplication (~7,700+ redundant lines)
- **Triplicated quantity kernels:** `quantity_ext.pyx`, `quantity_ext_openmp.pyx`, and `quantity_ext2.pyx` share ~90% code
- **Duplicate culvert classes:** `Culvert_operator` vs `Culvert_operator_Parallel` with near-identical logic
- **Parallel operator wrappers:** 5 files in `parallel/` that thin-wrap their `structures/` counterparts
- **Scattered utility functions:** `system_tools.py` (750 lines) and `numerical_tools.py` with overlapping helpers

### Linting
- **Zero infrastructure:** no linter, formatter, type checker, pre-commit hooks, or `.flake8`/`ruff.toml`
- **4,189 functions** with zero type annotations
- **Mixed conventions:** tabs vs spaces, wildcard imports, bare excepts, `os.system()` calls

### Testing
- **1,319 tests** (all `unittest.TestCase`), ~38 minutes wall time
- **Zero conftest.py**, zero fixtures, zero markers, zero parametrized tests
- **Isolation problems:** 7+ files write `domain.sww` to CWD, 47 `set_datadir('.')` calls, 198 `tempfile.mktemp()` uses (race conditions)
- **Coverage:** ~55% estimated, no `fail_under` enforced, no branch coverage
- **Validation tests:** only 5 of ~37 scenarios have automated `validate_*.py` scripts

### Build System
- Meson + meson-python (modern, good), but `setup.py` still present alongside
- C/Cython extensions compiled correctly, OpenMP support works
- No wheel CI — users must build from source

---

## Phase 0 — Test Infrastructure ("Refactor Without Fear")

**Goal:** Establish a test harness that catches regressions before any refactoring begins.

### 0.1 Fix Test Isolation (Week 1)

| Problem | Fix | Files |
|---------|-----|-------|
| 7+ tests write `domain.sww` to CWD | Use `tmp_path` fixture or `tempfile.mkdtemp()` | `test_*.py` across `shallow_water/`, `parallel/` |
| 47 × `set_datadir('.')` | Replace with `tmp_path` | grep for `set_datadir` |
| 198 × `tempfile.mktemp()` | Replace with `tempfile.mkstemp()` or `tmp_path` | grep for `mktemp` |
| Logging permanently disabled via `logging.disable(logging.CRITICAL)` | Use `caplog` fixture or scoped disable | `test_*.py` |
| Tests depend on execution order | Each test creates its own domain | Various |

### 0.2 Add Test Markers (Week 1)

```ini
# pyproject.toml additions
[tool.pytest.ini_options]
markers = [
    "slow: marks tests that take >10 seconds",
    "parallel: requires mpi4py",
    "gpu: requires CUDA/CuPy",
]
```

Tag the ~50 slowest tests with `@pytest.mark.slow`. This enables:
```bash
pytest -m "not slow"          # fast feedback (~5 min)
pytest -m slow                # full suite
```

### 0.3 Golden-Master Snapshots (Week 2)

Install `pytest-regressions` and create numerical snapshots for critical solvers:

```python
def test_evolve_bedslope(num_regression):
    """Golden-master: stage/xmom/ymom after 10 timesteps on bedslope problem."""
    domain = create_bedslope_domain(tmp_path)
    for _ in domain.evolve(yieldstep=0.1, finaltime=1.0):
        pass
    num_regression.check(
        {"stage": domain.quantities["stage"].centroid_values,
         "xmom": domain.quantities["xmomentum"].centroid_values},
        default_tolerance=dict(atol=1e-10, rtol=1e-10),
    )
```

Target: 8–12 golden-master tests covering the core solver paths (evolve, distribute, extrapolate, compute_fluxes).

### 0.4 Coverage Baseline & Enforcement (Week 2)

```ini
# .coveragerc updates
[run]
branch = true
source = anuga

[report]
fail_under = 55
show_missing = true

[html]
directory = htmlcov
```

Install `diff-cover` for PR enforcement:
```bash
diff-cover coverage.xml --compare-branch=origin/main --fail-under=80
```

This enforces 80% coverage on *changed lines only* without requiring a full-codebase coverage lift.

### 0.5 CI Test Matrix (Week 2)

Add a GitHub Actions workflow:

```yaml
# .github/workflows/tests.yml
strategy:
  matrix:
    python-version: ["3.10", "3.12", "3.13"]
    os: [ubuntu-latest]
jobs:
  test:
    steps:
      - run: pip install -e ".[dev]"
      - run: pytest -m "not slow" --tb=short -q
  test-slow:
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    steps:
      - run: pytest -m slow --tb=short -q
```

### Phase 0 Deliverables
- [ ] All tests pass in isolated `tmp_path` directories
- [ ] `pytest -m "not slow"` completes in <5 minutes
- [ ] 8–12 golden-master snapshots for core solver paths
- [ ] Coverage baseline established with `fail_under=55`
- [ ] `diff-cover` enforcing 80% on changed lines
- [ ] CI running on Python 3.10/3.12/3.13

---

## Phase 1 — Dependency Consolidation

**Goal:** Make `pip install anuga` actually work by declaring real dependencies and removing dead ones.

### 1.1 Fix pyproject.toml Dependencies (Week 3)

```toml
dependencies = [
    "numpy>=2.0.0",
    "scipy>=1.11",
    "netCDF4>=1.6",
    "matplotlib>=3.7",
    "meshpy>=2022.1",
    "dill>=0.3.7",
    "pymetis>=2023.1",
    "pyproj>=3.6",
    "affine>=2.4",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-regressions>=2.5",
    "diff-cover>=7.0",
    "ruff>=0.1",
]
parallel = [
    "mpi4py>=3.1",
]
```

### 1.2 Remove Dead Dependencies

| Dependency | Status | Action |
|-----------|--------|--------|
| cartopy | Never imported | Remove any install references |
| openpyxl | Never imported | Remove any install references |
| GDAL/osgeo | Partially removed | Complete removal (continue `remove-gdal` work) |
| `setup.py` | Superseded by meson | Delete after verifying `pip install .` works |

### 1.3 Fix Python Version Specifier

```toml
requires-python = ">=3.9, <3.14"
```

### Phase 1 Deliverables
- [ ] `pip install .` succeeds on a clean venv with all runtime deps
- [ ] `pip install ".[dev]"` provides test/lint tooling
- [ ] No phantom dependencies remain
- [ ] `setup.py` removed (meson-python is the build backend)

---

## Phase 2 — Linting & Code Quality

**Goal:** Establish automated code quality enforcement.

### 2.1 Add Ruff Configuration (Week 4)

```toml
# pyproject.toml additions
[tool.ruff]
target-version = "py39"
line-length = 120
src = ["anuga"]

[tool.ruff.lint]
select = [
    "E",    # pycodestyle errors
    "F",    # pyflakes
    "W",    # pycodestyle warnings
    "B",    # flake8-bugbear
    "I",    # isort
    "UP",   # pyupgrade
    "S",    # bandit (security)
]
ignore = [
    "E501",  # line length (enforce gradually)
]

[tool.ruff.lint.per-file-ignores]
"anuga/*/test_*.py" = ["S101"]  # assert is fine in tests
```

### 2.2 Pre-commit Hooks (Week 4)

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.3.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
```

### 2.3 Incremental Enforcement Strategy

1. **Week 4:** Run `ruff check --fix` once to auto-fix import sorting and simple issues
2. **Week 5:** Enable `ruff format` on changed files only (via pre-commit)
3. **Ongoing:** Add rules incrementally, never bulk-reformat the entire codebase (this would create massive merge conflicts with upstream)

> **Fork management note:** Bulk formatting is the single biggest source of merge conflicts. Always format only files you're already modifying.

### Phase 2 Deliverables
- [ ] `ruff check` passes (with agreed ignore list)
- [ ] Pre-commit hooks installed and documented
- [ ] CI runs `ruff check` on PRs

---

## Phase 3 — Code Deduplication & Modular Structure

**Goal:** Reduce ~7,700 lines of duplication and establish separation of concerns.

### 3.1 Unify Quantity Kernels (Week 6–7)

**Current state:** Three near-identical files:
- `quantity_ext.pyx` — serial
- `quantity_ext_openmp.pyx` — OpenMP-parallel
- `quantity_ext2.pyx` — legacy variant

**Target:** Single `quantity_ext.pyx` with compile-time OpenMP toggle:

```python
# meson.build
quantity_ext = py.extension_module(
    'quantity_ext',
    'quantity_ext.pyx',
    dependencies: use_openmp ? [openmp_dep] : [],
)
```

The OpenMP pragmas are no-ops when compiled without `-fopenmp`, so a single source file works for both paths.

### 3.2 Consolidate Parallel Operator Wrappers (Week 7)

**Current state:** 5 files in `parallel/` that are thin wrappers around `structures/`:
- `parallel_inlet_operator.py` wraps `inlet_operator.py`
- `parallel_boyd_box_operator.py` wraps `boyd_box_operator.py`
- etc.

**Target:** Move MPI-awareness into the base classes with a `self.parallel` flag, eliminating the wrapper files. The conditional import in `__init__.py` (lines 253–264) becomes unnecessary.

### 3.3 Merge Duplicate Culvert Classes (Week 8)

`Culvert_operator` and `Culvert_operator_Parallel` share ~85% code. Extract shared logic to base class, make parallel version a subclass override of `distribute()` only.

### 3.4 Clean Up Utility Modules (Week 8)

- Split `system_tools.py` (750 lines) into focused modules: `file_utils.py`, `env_utils.py`, `version_utils.py`
- Deduplicate `numerical_tools.py` vs scipy wrappers
- Move scattered geometry helpers into `geometry/`

### Phase 3 Deliverables
- [ ] Single quantity extension source (serial + OpenMP from one `.pyx`)
- [ ] Parallel operators consolidated (5 wrapper files removed)
- [ ] Culvert classes deduplicated
- [ ] Utility modules split and focused
- [ ] All golden-master tests still pass

---

## Phase 4 — Expanded Test Coverage

**Goal:** Lift meaningful coverage and fill gaps identified during refactoring.

### 4.1 Modernise Test Patterns (Week 9)

- Convert key test classes from `unittest.TestCase` to plain pytest functions where it simplifies the code
- Add `conftest.py` with shared fixtures:

```python
# anuga/conftest.py
import pytest
import tempfile, os

@pytest.fixture
def domain(tmp_path):
    """Provides a small rectangular domain for testing."""
    from anuga import rectangular_cross_domain
    domain = rectangular_cross_domain(5, 5, len1=10, len2=10)
    domain.set_datadir(str(tmp_path))
    domain.set_name("test")
    return domain
```

### 4.2 Integrate Validation Tests (Week 9)

Wire the 5 existing `validate_*.py` scripts into pytest:

```python
# validation_tests/conftest.py
import subprocess, pytest

def pytest_collect_file(parent, file_path):
    if file_path.name.startswith("validate_") and file_path.suffix == ".py":
        return pytest.Module.from_parent(parent, path=file_path)
```

### 4.3 Coverage Targets (Week 10)

| Module | Current (est.) | Target | Priority |
|--------|---------------|--------|----------|
| `shallow_water/` | ~65% | 80% | High — core solver |
| `fit_interpolate/` | ~50% | 70% | High — data pipeline |
| `file_conversion/` | ~40% | 60% | Medium |
| `structures/` | ~55% | 75% | High — culverts/inlets |
| `geometry/` | ~70% | 85% | Low — already decent |

### Phase 4 Deliverables
- [ ] Shared `conftest.py` with `domain` fixture
- [ ] Validation tests discoverable by pytest
- [ ] Coverage `fail_under` raised to 65%
- [ ] `diff-cover` threshold maintained at 80% for new code

---

## Fork/Upstream Management Strategy

### Current Divergence Analysis

| Branch | Status |
|--------|--------|
| `main` | **Perfectly in sync** with `upstream/main` (0 commits divergence) |
| `claude-experiments` | 55 commits ahead (GDAL removal + subgrid terrain sampling) |
| `feature/subgrid-terrain-sampling` | 37 commits ahead |
| `remove-gdal` | 40 commits ahead |
| `upstream develop` | **116 commits ahead** of `upstream/main` — stoiver actively refactoring evolve loop |

### Critical Overlap Zones

These 10 files are modified by **both** Hydrata branches and upstream `develop`:

| File | Risk | Upstream Change | Hydrata Change |
|------|------|-----------------|----------------|
| `shallow_water_domain.py` | **HIGH** | Evolve loop refactor, centroid distribution | GDAL removal, subgrid hooks |
| `sw_domain_openmp_ext.pyx` | **HIGH** | Centroid-based extrapolation | OpenMP changes |
| `sw_domain.h` | **HIGH** | New struct fields | Subgrid fields |
| `generic_domain.py` | Medium | Evolve refactor | Minor edits |
| `quantity.py` | Medium | Centroid distribution | Subgrid sampling |
| `fit.py` | Low | Bug fixes | GDAL removal |
| `asc2dem.py` | Low | Minor | GDAL→rasterio |
| `dem2pts.py` | Low | Minor | GDAL→rasterio |
| `sww2dem.py` | Low | Minor | GDAL→rasterio |
| `pmesh2domain.py` | Low | Minor | Minor |

### Branching Strategy

```
upstream/main ──────────────────────────────────────────────▶
     │
     ├── origin/main (kept in sync, never diverge)
     │       │
     │       ├── anuga-4.0-refactor-plan (this plan + Phase 0–4 work)
     │       │
     │       ├── remove-gdal (GDAL→rasterio, candidate for upstream PR)
     │       │
     │       └── feature/subgrid-terrain-sampling (Hydrata-specific)
     │
upstream/develop ──────────────────────────────────────────▶
     (116 commits ahead — evolve loop refactor by stoiver)
```

### Rules for Fork Hygiene

1. **Keep `main` in sync.** Regularly `git fetch upstream && git merge upstream/main` into `origin/main`. This is already the case — maintain it.

2. **Rebase feature branches on `main`, not `develop`.** Upstream `develop` is unstable and moves fast. Feature branches should track `main` for stability.

3. **Upstream-worthy changes as separate PRs.** These changes benefit the whole community and should be submitted to `anuga-community/anuga_core`:

   | Change | Branch | PR Priority |
   |--------|--------|-------------|
   | Dependency declaration fixes | New branch from `main` | **Immediate** — uncontroversial, everyone benefits |
   | Test isolation fixes | New branch from `main` | **High** — improves CI for everyone |
   | `setup.py` removal | New branch from `main` | **Medium** — coordinate with stoiver |
   | Ruff/linting config | New branch from `main` | **Medium** — needs buy-in |
   | GDAL→rasterio | `remove-gdal` | **After** upstream discuss — big change |
   | Quantity kernel unification | New branch | **After** coordinate with samcom12 GPU work |

4. **Never bulk-format files you're not otherwise changing.** This is the #1 source of merge conflicts between forks. Only format files touched by your feature work.

5. **Monitor upstream `develop`.** Stoiver's evolve loop refactor (PR #115) will eventually merge to `main`. When it does:
   - Rebase `claude-experiments` and `feature/subgrid-terrain-sampling` onto the new `main`
   - Resolve conflicts in the 3 HIGH-risk files (`shallow_water_domain.py`, `sw_domain_openmp_ext.pyx`, `sw_domain.h`)
   - Re-run golden-master tests to verify numerical equivalence

6. **Coordinate with samcom12 (GPU fork) on C/Cython changes.** Their quantity kernel work and our unification effort (Phase 3.1) touch the same files. A shared strategy avoids duplicated work.

### Sync Cadence

| Action | Frequency |
|--------|-----------|
| `git fetch upstream` | Weekly |
| Merge `upstream/main` → `origin/main` | When upstream main advances |
| Check upstream `develop` for conflicts | Before starting any Phase 3 C/Cython work |
| Rebase feature branches on `main` | After each upstream merge |

---

## Timeline Summary

| Phase | Scope | Duration | Depends On |
|-------|-------|----------|------------|
| **Phase 0** | Test infrastructure | Weeks 1–2 | Nothing |
| **Phase 1** | Dependencies | Week 3 | Phase 0 |
| **Phase 2** | Linting | Weeks 4–5 | Phase 0 |
| **Phase 3** | Deduplication | Weeks 6–8 | Phases 0, 1, 2 |
| **Phase 4** | Test coverage | Weeks 9–10 | Phases 0, 3 |

Phases 1 and 2 can run in parallel after Phase 0 is complete.

---

## Decisions Log

| Decision | Rationale |
|----------|-----------|
| **No mutation testing** | Poor ROI: can't mutate Cython, tolerance-based assertions kill most mutants, 50K+ mutants would take days |
| **No property-based testing (yet)** | Valuable but not a prerequisite for refactoring — defer to Phase 4 |
| **`pytest-regressions` for golden masters** | Purpose-built for numpy array tolerance comparisons, stores baselines as files in VCS |
| **`diff-cover` over full coverage targets** | Pragmatic — enforces quality on new/changed code without requiring massive backfill |
| **Ruff over flake8/black** | Single tool replaces linter + formatter + import sorter, 10–100x faster |
| **Incremental formatting** | Bulk reformatting would create merge conflicts with upstream and GPU fork |
| **Upstream PRs for generic improvements** | Dependency fixes, test isolation, and linting benefit the whole community |
