# Quickstart

## Install

```bash
pip install "hyperwave[plot,sampling]"
```

## A complete fast PE run

The 2-parameter "fast PE" demo recovers chirp mass and mass ratio with the
hyperbolic likelihood in minutes on a laptop. The full script is
`examples/pe_fast/bbh_fast_pe.py`; the essential steps:

```python
import bilby
import numpy as np
from hyperwave.detectors.lvk import GW, DetectorNoise
from hyperwave.likelihoods import GWLikelihoods
from hyperwave.inference import LVKinference

# 1. data: Gaussian noise + a BBH injection
trigger = 1268189526.951953
names = ["chirp_mass", "mass_ratio"]
theta_true = [28.1, 0.806]

noise = DetectorNoise(4.0, 2048.0, trigger, ["H1", "L1"],
                      minimum_frequency=20.0, maximum_frequency=512.0)
noise.generate_noise(real_noise=False, seed=42)
template = GW(noise, approximant="IMRPhenomPv2", parameters=names,
              static_parameters={...})           # fix the extrinsic parameters
template.make_injections_to_ifo(theta_true)

# 2. likelihood on the masked band
f, asd0 = template.detector_asd_masked(0)
psd = np.array([asd0**2, template.detector_asd_masked(1)[1]**2])
data = np.array([template.detector_data_fd(0), template.detector_data_fd(1)])
like = GWLikelihoods(data=data, f=f, ifos_list=["H1", "L1"], noise=psd,
                     template=template, ddims=False, nsegs=2)

# 3. priors: science parameters + hyperbolic shape parameters
priors = bilby.core.prior.PriorDict()
priors["chirp_mass"] = bilby.gw.prior.UniformInComponentsChirpMass(25, 31)
priors["mass_ratio"] = bilby.gw.prior.UniformInComponentsMassRatio(0.5, 1.0)
noise_priors = {r"$\alpha$": bilby.core.prior.Uniform(0, 30),
                r"$\delta_0$": bilby.core.prior.Uniform(0, 30),
                r"$\delta_1$": bilby.core.prior.Uniform(0, 30)}

# 4. sample (eryn here; sampler_name="pocomc" also works)
inf = LVKinference(like.hyperbolic_classic, sampler_name="eryn",
                   priors=priors, noise_priors=noise_priors,
                   common_params={"save_dir": "out", "TAG": "fast", "like": "hyperbolic"},
                   sampler_kwargs=dict(nwalkers=32, ntemps=8, burn=2000, nsteps=8000))
inf.run()

# 5. results
result = inf.get_result(injection=theta_true, parameter_names=names)
print(result.median(), result.credible_interval())
result.corner()                       # corner plot
result.save("out/fast_pe.h5")
```

## Where to go next

- Full 14/15-parameter PE: `examples/pe_full/bbh_full_pe.py`
- Real event (GW150914): `examples/real_event/gw150914_pe.py`
- The four likelihoods: [Likelihoods](likelihoods.md)
- GPU waveforms: [GPU acceleration](gpu.md)
- Posterior calibration: [Validation](validation.md)
