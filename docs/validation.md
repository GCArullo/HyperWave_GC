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

The standalone PP-test helper module is not bundled in this source snapshot.
Use `Result.credible_level(...)` to compute per-run credible levels, then pass
those arrays to your project-level PP-test implementation.

## Campaigns

Campaign orchestration is also left to downstream analysis scripts in this
snapshot. The heterodyne likelihood and ml4gw GPU waveforms are the main levers
for making large PP campaigns cheaper.
