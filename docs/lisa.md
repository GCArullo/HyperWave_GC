# LISA

HyperWave runs LISA parameter estimation through an A/E(/T) bridge that adapts
any TDI waveform generator to the standard likelihood stack:

```
generator (bbhx / GBGPU)  →  LISAAETTemplate  →  build_lisa_aet_likelihood  →  sampler
```

The bridge accepts raw NumPy arrays or lisatools containers
(`AnalysisContainer` / `DataResidualArray`; imported lazily), and exposes the
**vectorized** path: pass `batch_signal_model` and the whole walker population
is generated in one bbhx/GBGPU call per likelihood evaluation.

## Massive black-hole binaries (SMBHB, bbhx)

```bash
python examples/lisa/smbhb_bbhx_pe.py --sampler both --quick   # eryn + pocomc timing
python examples/lisa/smbhb_bbhx_pe.py --sampler eryn --steps 40000
```

11 source parameters + hyperbolic shape parameters, A/E channels, PhenomD
(2,2). The example calibrates the analytic PSD stub to a target SNR
(`--target-snr`).

## Galactic binaries (UCB, GBGPU)

```bash
python examples/lisa/ucb_gbgpu_pe.py --sampler eryn --steps 40000
```

GBGPU returns each source's narrow band; the example scatters evaluations onto
a fixed global grid (window centred on the injected `f0`, `df = 1/Tobs`) so the
likelihood residual is well defined. 8 sampled parameters; `fddot` static.

## Building the LISA stack

!!! danger "Do not `pip install bbhx`"
    The PyPI wheel (1.2.3) crashes on every call (an orbits-pointer bug fixed
    upstream only in unreleased v1.2.5) **and** is AVX-512-compiled, which
    SIGILLs on AMD nodes. Build from source — the verified recipe (four
    packages, ~15 min) is in `ENVIRONMENT.md` at the repository root.

`gbgpu` 1.1.3 from PyPI works where the CPU has AVX-512 (Intel Skylake) or via
its CUDA path; its modern (`master`) API requires a migration tracked in
`TODO.md`.

## Custom generators

Any callable returning `(nchannels, nfreq)` — or `(N, nchannels, nfreq)` for
the batched path — plugs in directly:

```python
from hyperwave.detectors.lisa import LISAAETTemplate, build_lisa_aet_likelihood

template = LISAAETTemplate(
    parameters=["m1", "m2", ...],
    signal_model=my_generator,            # kwargs -> (2, nfreq)
    batch_signal_model=my_batch_generator,  # arrays -> (N, 2, nfreq)
    channels=("A", "E"),
)
like = build_lisa_aet_likelihood(data=data, template=template,
                                 sensitivity=psd, freqs=freqs,
                                 channels=("A", "E"), ddims=False, nsegs=2)
```
