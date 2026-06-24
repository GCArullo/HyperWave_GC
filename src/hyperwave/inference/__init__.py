"""Inference helpers for HyperWave."""

from .convergence import WaveletConvergenceStopping
from .sampling import DataInference, LVKinference


def _missing_optional(name, module):
    def missing(*args, **kwargs):
        raise ImportError(f"{name} requires hyperwave.inference.{module}, which is unavailable.")

    missing.__name__ = name
    return missing


# Optional submodules — re-exported only if their source files are present.
# (wavelet_proposals, fastjumps, flow_proposals, wavelet_priors are pending upload.)
try:
    from .wavelet_proposals import (
        DataInformedMarginal,
        MatchedFilterBirth,
        WaveletFisherMove,
        WaveletHalfCycleMove,
        WaveletSkyRingMove,
        build_flow_proposal,
        build_guided_birth,
        build_mf_birth,
        guided_initial_wavelets,
    )
except ImportError:
    DataInformedMarginal = _missing_optional("DataInformedMarginal", "wavelet_proposals")
    MatchedFilterBirth = _missing_optional("MatchedFilterBirth", "wavelet_proposals")
    WaveletFisherMove = _missing_optional("WaveletFisherMove", "wavelet_proposals")
    WaveletHalfCycleMove = _missing_optional("WaveletHalfCycleMove", "wavelet_proposals")
    WaveletSkyRingMove = _missing_optional("WaveletSkyRingMove", "wavelet_proposals")
    build_flow_proposal = _missing_optional("build_flow_proposal", "wavelet_proposals")
    build_guided_birth = _missing_optional("build_guided_birth", "wavelet_proposals")
    build_mf_birth = _missing_optional("build_mf_birth", "wavelet_proposals")
    guided_initial_wavelets = _missing_optional("guided_initial_wavelets", "wavelet_proposals")

try:
    from .fastjumps import FastJumpInference, FastJumpModel, FastJumpResult, PocoFastJumps
except ImportError:
    FastJumpInference = _missing_optional("FastJumpInference", "fastjumps")
    FastJumpModel = _missing_optional("FastJumpModel", "fastjumps")
    FastJumpResult = _missing_optional("FastJumpResult", "fastjumps")
    PocoFastJumps = _missing_optional("PocoFastJumps", "fastjumps")

try:
    from .flow_proposals import (
        AdaptiveFlowProposal,
        ContextAwareBirthRJMove,
        FlowFitReport,
        FlowTrainingCallback,
        flow_backend_available,
        make_flow_distribution_move,
        make_flow_rj_move,
    )
except ImportError:
    AdaptiveFlowProposal = _missing_optional("AdaptiveFlowProposal", "flow_proposals")
    ContextAwareBirthRJMove = _missing_optional("ContextAwareBirthRJMove", "flow_proposals")
    FlowFitReport = _missing_optional("FlowFitReport", "flow_proposals")
    FlowTrainingCallback = _missing_optional("FlowTrainingCallback", "flow_proposals")
    make_flow_distribution_move = _missing_optional("make_flow_distribution_move", "flow_proposals")
    make_flow_rj_move = _missing_optional("make_flow_rj_move", "flow_proposals")

    def flow_backend_available():
        return False


try:
    from .wavelet_priors import CosinePrior, SNRPrior, build_wavelet_priors
except ImportError:
    CosinePrior = _missing_optional("CosinePrior", "wavelet_priors")
    SNRPrior = _missing_optional("SNRPrior", "wavelet_priors")
    build_wavelet_priors = _missing_optional("build_wavelet_priors", "wavelet_priors")

InferenceRunner = LVKinference

__all__ = [
    "LVKinference",
    "InferenceRunner",
    "DataInference",
    "FastJumpInference",
    "FastJumpModel",
    "FastJumpResult",
    "PocoFastJumps",
    "AdaptiveFlowProposal",
    "ContextAwareBirthRJMove",
    "FlowFitReport",
    "FlowTrainingCallback",
    "flow_backend_available",
    "make_flow_distribution_move",
    "make_flow_rj_move",
    # wavelet reconstruction
    "build_wavelet_priors",
    "SNRPrior",
    "CosinePrior",
    "WaveletConvergenceStopping",
    "build_guided_birth",
    "build_mf_birth",
    "WaveletFisherMove",
    "WaveletHalfCycleMove",
    "WaveletSkyRingMove",
    "build_flow_proposal",
    "guided_initial_wavelets",
    "DataInformedMarginal",
    "MatchedFilterBirth",
]
