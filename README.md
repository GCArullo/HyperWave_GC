<p align="center">
  <img src="hyperwave/static/logo.png" alt="HyperWave logo" height="260" />
</p>

# HyperWave

HyperWave is a Python package for robust gravitational-wave inference
with hyperbolic likelihoods. It combines detector/noise utilities,
waveform-facing likelihood code, inference helpers, and plotting tools
for studying non-Gaussian noise and glitch-tolerant parameter
estimation.

## What the package provides

-   Hyperbolic and Gaussian likelihoods for GW data analysis\
-   Frequency-domain utilities for data-only noise studies\
-   LVK-oriented detector and waveform helpers\
-   Eryn and pocoMC based inference helpers\
-   Plotting utilities and example notebooks for common workflows

## Installation

### 1. Create a clean environment

``` bash
conda create -n hyperwave python=3.11 -y
conda activate hyperwave
python -m pip install -U pip setuptools wheel
```

### 2. Install HyperWave

``` bash
pip install .
```

For development:

``` bash
pip install -e .
```

### 3. Optional extras

``` bash
pip install -e ".[plot]"
pip install -e ".[sampling]"
pip install -e ".[dev]"
```

### 4. GPU support (CUDA 12.x)

``` bash
pip install -e ".[gpu]"
```

### HPC users

``` bash
srun --partition=gpu --gres=gpu:1 --pty bash
conda activate hyperwave
nvidia-smi
```

### Verify GPU

``` bash
python - << 'EOF'
import cupy as cp
print(cp.cuda.is_available())
EOF
```

## License

MIT License
