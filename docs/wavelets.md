# Wavelet reconstruction

Model-agnostic signal reconstruction with Morlet–Gabor wavelets and
reversible-jump MCMC (Eryn), in the spirit of BayesWave: the number of wavelets
is itself sampled, an SNR prior supplies the Occam penalty, and an optional
extrinsic branch samples the sky position ($\mathrm{ra}$, $\mathrm{dec}$,
$\psi$, ellipticity).

The current source tree contains the likelihood-side interface, but does not
bundle the wavelet template/proposal implementation or a standalone
reconstruction driver script. Missing wavelet helpers raise `ImportError` with
an explicit message when called.
