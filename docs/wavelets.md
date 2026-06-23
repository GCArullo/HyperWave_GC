# Wavelet reconstruction

Model-agnostic signal reconstruction with Morlet–Gabor wavelets and
reversible-jump MCMC (Eryn), in the spirit of BayesWave: the number of wavelets
is itself sampled, an SNR prior supplies the Occam penalty, and an optional
extrinsic branch samples the sky position ($\mathrm{ra}$, $\mathrm{dec}$,
$\psi$, ellipticity).

```bash
python examples/bbh_wavelet_reconstruction.py \
    --proposal mffisher --sample-sky \
    --nwalkers 50 --ntemps 10 --nsteps 30000 --burn 10000 --device gpu
```