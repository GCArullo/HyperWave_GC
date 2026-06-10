# Contributing to HyperWave

Thanks for your interest in improving HyperWave!

## Development setup

```bash
git clone https://github.com/asasli/HyperWave.git
cd HyperWave
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]
# lalsuite/gwpy are easiest via conda: conda install -c conda-forge lalsuite gwpy
pre-commit install
```

## Workflow

- Format and lint with `ruff`:
  ```bash
  ruff check . && ruff format .
  ```
- Run the tests:
  ```bash
  pytest
  ```
  GW tests skip automatically without `lalsuite`; ml4gw tests skip without
  `ml4gw`/`torch` (note: `ml4gw` requires Python < 3.13).

## Numerical agreement

HyperWave is validated against bilby as a reference. When you touch the waveform or detector layer, keep the existing guarantees:

- `LALWaveform` must remain **bit-exact** vs
  `bilby.gw.source.lal_binary_black_hole`.
- The batched likelihood must equal the per-walker path bit-for-bit.

Add or update a test in `tests/` for any behavioural change, and mention the measured agreement in the PR description.

## Scope notes

- The `inference` layer keeps bilby for its prior distributions. This should change in the future.
- The `detectors`/`waveforms`/`likelihoods` layers are bilby-free.

## Pull requests

Keep PRs focused, describe the motivation, and ensure `ruff check` and `pytest` pass. Open an issue first for larger or architectural changes.
