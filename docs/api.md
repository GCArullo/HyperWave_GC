# API reference

## Likelihoods

::: hyperwave.likelihoods.GWLikelihoods

::: hyperwave.likelihoods.HeterodyneLikelihood

::: hyperwave.likelihoods.heterodyne.heterodyne_bin_edges

::: hyperwave.likelihoods.WaveletLikelihood

## Inference

::: hyperwave.inference.LVKinference

The wavelet-proposal API (`MatchedFilterBirth`, `build_mf_birth`,
`build_guided_birth`, ...) is optional and raises `ImportError` with a clear
message when the implementation module is unavailable.

## Results

::: hyperwave.result.Result

The PP-test validation helper module is not bundled in this source snapshot.
Use `Result.credible_level(...)` for per-run credible levels.

## Detectors & templates

::: hyperwave.detectors.lvk.GW

::: hyperwave.detectors.lisa.aet
