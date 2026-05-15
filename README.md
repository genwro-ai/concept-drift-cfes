# concept_drift_cfes

Code for the paper *Counterfactual Explanations Under Concept Drift* (IJCAI-ECAI 2026).

A lightweight, model-agnostic maintenance scheme that repairs existing counterfactual explanations (CFEs) as an online classifier is repeatedly updated under concept drift, using local black-box sampling to preserve validity and plausibility.

## Setup

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Scripts

### Quality experiments

Runs the main quality evaluation (validity, proximity, plausibility) across synthetic drifting streams, online classifiers, and CFE generators. Results are written to `artifacts/paper_runs/`.

```bash
uv run python scripts/run_paper_experiments.py --repeats 5
```

### Runtime comparison

Measures wall-clock time for each maintenance variant and reference method. Results are written to `artifacts/paper_timing/`.

```bash
uv run python scripts/run_paper_timing.py
```

## Contents

- [src/concept_drift_cfes/update.py](src/concept_drift_cfes/update.py) — proposed update methods (validity-plausibility and plausibility-only)
- [src/concept_drift_cfes/reference/](src/concept_drift_cfes/reference/) — reference CFE generators (Growing Spheres, RobX)
- [scripts/run_paper_experiments.py](scripts/run_paper_experiments.py) — main quality experiments
- [scripts/run_paper_timing.py](scripts/run_paper_timing.py) — runtime comparison
