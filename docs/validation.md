# Validation (PP tests)

A pipeline you can cite needs calibration evidence: if you inject from the
prior and recover with the pipeline, the credible level of each true value must
be uniform — the PP plot hugs the diagonal.

## Result objects

Every run can produce a `hyperwave.Result`:

```python
result = inf.get_result(injection=theta_true, parameter_names=names)
result.median();  result.credible_interval(0.9)
result.credible_level(theta_true)    # per-parameter quantile of the truth
result.corner();  result.save("run.h5")
```

## PP machinery

```python
from hyperwave.validation import credible_levels, pp_pvalues, make_pp_plot

levels = credible_levels(results)          # (n_runs, n_params)
combined_p, per_param = pp_pvalues(levels, names)   # KS per param + Fisher combine
fig = make_pp_plot(results)                # with binomial confidence bands
```

The unit tests demonstrate the machinery's sensitivity: a calibrated pipeline
passes (combined `p > 0.05`), while biased or overconfident posteriors are
detected (`p < 0.01`).

## Campaigns

`hyperwave.validation.run_pp_campaign` loops injections, saves each `Result`,
and writes `pp_summary.json` + `pp_plot.png`; it is resumable (skips completed
injections). A concrete fast campaign lives in `examples/validation/pp_fast.py`
with a cluster runner in `examples/clusters/pp_fast.slurm`.

!!! note
    The full 100-injection hyperbolic-likelihood campaign is pending the
    GPU/parallel throughput work (`TODO.md`) — the heterodyne likelihood and
    ml4gw GPU waveforms are the levers that make it cheap.
