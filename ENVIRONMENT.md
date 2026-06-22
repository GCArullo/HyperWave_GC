# HyperWave environments

HyperWave runs in a conda environment. Three are used on the MSI cluster:

| Env | Purpose | Stack |
|---|---|---|
| `hyperwave` | CPU-only GW pipeline | lalsuite, gwpy, eryn, pocomc |
| `hyperwave-gpu` | GPU GW pipeline (wavelets, CBC PE) | + cupy (CUDA 11.x) |
| **`hyperwave-dev`** | **GW + LISA** (everything in one env) | + bbhx, gbgpu, lisatools |

This document reproduces **`hyperwave-dev`** — the GPU GW stack *plus* the LISA
waveform stack (Michael Katz's bbhx / gbgpu / lisaanalysistools), so the
ground-based and LISA examples both run from a single environment.

---

## Prerequisites

- `miniconda3` at `/users/4/asasli/miniconda3` (adjust paths below if different).
- A working **`hyperwave-gpu`** env (the GW pipeline base).
- A GPU node with the cluster's **CUDA 11.8** toolkit module
  (`cuda/11.8.0-gcc-7.2.0-xqzqlf2`), used to build/run the LISA CUDA stack.
- The LISA packages ship prebuilt wheels (numpy 2.x / numba), so no compilation
  is normally needed — but they require a recent CPU (AVX2); see *Known issues*.

---

## Build steps

### 1. Clone the GW environment (login node)

`hyperwave-dev` starts as a clone of the validated GW env, so the full
lal/eryn/pocomc/cupy stack is preserved:

```bash
source /users/4/asasli/miniconda3/etc/profile.d/conda.sh
conda create --name hyperwave-dev --clone hyperwave-gpu --yes
```

### 2. Add the LISA stack (GPU node, CUDA 11.8)

The LISA packages are installed on a GPU node with the CUDA toolkit on PATH.
Submit the provided build job:

```bash
sbatch examples/clusters/build_hyperwave_dev.slurm
```

or, interactively:

```bash
srun --partition=a100-4 --gres=gpu:a100:1 --cpus-per-task=8 --mem=32G \
     --time=01:00:00 --account=vuk --pty bash
source /users/4/asasli/miniconda3/etc/profile.d/conda.sh
module load cuda/11.8.0-gcc-7.2.0-xqzqlf2
export CUDA_HOME=/common/software/install/spack/linux-centos7-ivybridge/gcc-7.2.0/cuda-11.8.0-xqzqlf2v77opht3bv4onsqt7uuiomqec
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
conda activate hyperwave-dev

conda install -y -c conda-forge gsl lapack fftw liblapacke
pip install gbgpu lisaanalysistools
```

> **Do NOT `pip install bbhx`.** The PyPI wheel (1.2.3) is broken twice over:
> its compiled response kernel takes the lisatools orbits object as a C pointer
> (`TypeError: an integer is required` on every call — fixed upstream only in
> the unreleased v1.2.5), and the wheel is AVX-512-compiled (SIGILLs on the AMD
> Milan nodes). Build from source instead (verified recipe, login node, ~15 min):
>
> ```bash
> conda activate hyperwave-dev
> pip install scikit-build-core setuptools_scm ninja cython pybind11
> pip install "git+https://github.com/mikekatz04/gpubackendtools.git"
> # fix an include-guard collision with bbhx's own global.h:
> sed -i '1,3s/__GLOBAL_H__/__GBT_GLOBAL_H__/' \
>   $CONDA_PREFIX/lib/python3.10/site-packages/gpubackendtools/cutils/gbt_global.h
> export CFLAGS="-march=haswell -mtune=generic" CXXFLAGS="$CFLAGS" FFLAGS="-march=haswell"
> export CMAKE_ARGS="-DLISATOOLS_LAPACKE_FETCH=ON -DLISATOOLS_MARCH=haswell -Dpybind11_DIR=$(python -m pybind11 --cmakedir)"
> pip install --no-build-isolation "git+https://github.com/mikekatz04/LISAanalysistools.git"
> # uncomment `class AddOrbits` in site-packages/lisatools/cutils/Detector.hpp
> # (upstream ships it commented out; bbhx's Response.hh inherits from it; also
> #  initialise `Orbits *orbits = nullptr;`), then:
> export GBT=$CONDA_PREFIX/lib/python3.10/site-packages/gpubackendtools/cutils
> export CFLAGS="$CFLAGS -I$GBT" CXXFLAGS="$CXXFLAGS -I$GBT"
> export CMAKE_ARGS="-DBBHX_LAPACKE_FETCH=ON -DBBHX_MARCH=haswell -Dpybind11_DIR=$(python -m pybind11 --cmakedir)"
> pip install --no-build-isolation "git+https://github.com/mikekatz04/BBHx.git"
> ```
>
> `-march=haswell` (AVX2 baseline; gcc 8.5 does not know `x86-64-v3`) runs on
> both the AMD Milan A100 nodes and the Intel Skylake v100 nodes. A CPU-only
> build registers no CUDA backends, and gpubackendtools-main auto-picks
> `bbhx_cuda13x` and crashes — construct with `force_backend="cpu"` (the SMBHB
> example does this automatically).

Versions verified working: source-built `bbhx 1.2.5.post1.dev2`,
`lisaanalysistools 1.2.8.post1.dev5`, `gpubackendtools 0.1.1.post1.dev6`,
plus wheel `gbgpu 1.1.3` (alongside `cupy-cuda12x 14.1.1`, `lal 7.7.1`,
`eryn 1.2.6`, `pocomc 1.2.6`, `numpy 2.2.6`). Note: gbgpu 1.1.3's CPU kernel is AVX-512
(v100/Skylake only, or `--use-gpu`); its modernization is tracked in TODO.md.

### 3. Verify

```bash
for m in cupy lal eryn pocomc bbhx gbgpu lisatools; do
    python -c "import $m" && echo "OK $m" || echo "FAIL $m"
done
```

All seven import cleanly. The HyperWave package itself
(`python -c "import hyperwave"` with `PYTHONPATH=$PWD/src`) and the LISA
examples are run on a GPU node (see below).

---

## Running on a GPU node

For CPU runs no CUDA module is needed. For the LISA **GPU** path load
`cuda/12.1.1` and set the library paths as in the CUDA section below, then:

```bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
# Full end-to-end PE: generator -> LISAAETTemplate -> build_lisa_aet_likelihood -> Eryn
python examples/lisa/smbhb_bbhx_pe.py --quick   # SMBHB (bbhx) A/E + hyperbolic PE
python examples/lisa/ucb_gbgpu_pe.py  --quick   # galactic binary (gbgpu) A/E + hyperbolic PE
python examples/lisa/smbhb_bbhx_pe.py --snr-only # waveform-generation / SNR check only
```

Both examples run the *same* hyperbolic `GWLikelihoods` / `LVKinference` stack as
the ground-based runs, via the A/E bridge in `hyperwave.detectors.lisa`. Or submit
the batch job (SNR check + quick PE for both sources):

```bash
sbatch examples/clusters/lisa_pe.slurm                              # quick PE
sbatch --export=ALL,MODE=prod,STEPS=40000 examples/clusters/lisa_pe.slurm   # production
```

The bridge + sampler wiring is validated on CPU (no LISA waveforms needed) by
`python examples/lisa/_wiring_smoke.py`. The waveform generators (bbhx/gbgpu)
themselves need the GPU node.

---

## Known issues

- **[RESOLVED by the source builds]** The PyPI lisatools wheel's
  `AnalysisContainer` used to SIGILL on the AMD Zen3 nodes (AVX-512 wheel); the
  source-built lisatools (`-march=haswell`, recipe above) imports and runs
  everywhere. The A/E bridge still imports lisatools lazily — harmless, and it
  keeps the bridge importable in minimal envs without the LISA stack.
- **`fastlisaresponse` crashes on import** (`Illegal instruction`). It is **not
  required** for the SMBHB (bbhx) or UCB (gbgpu) examples — both packages carry
  their own LISA response — so it is omitted. Skip it unless you need the
  generic time-domain response (e.g. EMRIs).
- **Intermittent `Illegal instruction` on some GPU nodes.** The prebuilt LISA
  wheels (and their numba/numpy deps) use AVX2; a few older/odd nodes trip a
  SIGILL during the compiled *waveform* kernels. Mitigations: prefer the
  **GPU** path (`use_gpu=True`), and if a node SIGILLs, resubmit (the scheduler
  lands on a capable node). Imports were verified on `aga06`/`aga20`/`aga29`.
- **numpy is 2.x** in this env. HyperWave is numpy-2-aware (it shims the removed
  `np.in1d`), and the GW stack imports fine; keep numpy ≥ 2 (the LISA wheels
  require it).

---

## Quick reference: activate + run

```bash
source /users/4/asasli/miniconda3/etc/profile.d/conda.sh
module load cuda/11.8.0-gcc-7.2.0-xqzqlf2
export CUDA_HOME=/common/software/install/spack/linux-centos7-ivybridge/gcc-7.2.0/cuda-11.8.0-xqzqlf2v77opht3bv4onsqt7uuiomqec
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export PYTHONPATH="/users/4/asasli/HyperwaveV2/src:$PYTHONPATH"
conda activate hyperwave-dev
```

---

## CUDA (A100) build of the LISA stack — verified recipe

The CPU recipe above, plus (on an A100 node):

```bash
module load cuda/12.1.1            # nvcc >= 12 is REQUIRED (11.8 rejects the CUDA source)
export CUDA_HOME=/common/software/install/manual/cuda/12.1.1
export PATH="$CUDA_HOME/bin:$PATH"
export CUDAFLAGS="-I$CONDA_PREFIX/lib/python3.10/site-packages/gpubackendtools/cutils"
pip install cupy-cuda12x nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 \
            nvidia-nvjitlink-cu12 nvidia-cusparse-cu12 nvidia-cuda-nvrtc-cu12
# rebuild lisatools THEN bbhx with --no-cache-dir (pip's wheel cache serves
# stale same-version builds); re-apply the AddOrbits patch after the lisatools
# reinstall. See examples/clusters/build_lisa_gpu_a100.slurm for the full job.
```

Run-time requirements: the cuda module's `lib64` (or the pip `nvidia/*/lib`
dirs) on `LD_LIBRARY_PATH`; **CuPy frequency arrays in, CuPy waveforms out**
(`.get()` to NumPy); orbits constructed with the **same** `force_backend` as
the response. Verified: SMBHB PE on cuda12x, 3200 samples / 57.9 s (A100);
batch-128 waveform generation 6.3× over the node's CPU.
