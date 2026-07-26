"""MBE neuron package: Multi-Basis Exponential Decay neuron and fitting tools."""
from .neuron import MBENeuron, MBEConfig
from .signed import SignedMBENeuron
from .surrogate import heaviside
from . import functions
from . import spiking_ops
from .fit import fit_function, fit_model, solve_readout, FitResult
# Standalone PASN (successive-approximation) neuron.
from .pasn import PrefixRouter, PASNNeuron, build_pasn, sar_encode
# MBE-PASN (the MBE-improvement variant: prefix banks of MBE neurons).
from .mbe_pasn import MBEPASNNeuron, build_mbe_pasn
# MBE-PASN-S (shared MBE basis set, prefix-routed readout).
from .mbe_pasn_s import MBEPASNSNeuron, build_mbe_pasn_s
# Unified cost metrics across all neuron types.
from .metrics import neuron_cost, spikes_per_input

__all__ = [
    "MBENeuron",
    "MBEConfig",
    "SignedMBENeuron",
    "heaviside",
    "functions",
    "spiking_ops",
    "fit_function",
    "fit_model",
    "solve_readout",
    "FitResult",
    # standalone PASN
    "PrefixRouter",
    "PASNNeuron",
    "build_pasn",
    "sar_encode",
    # MBE-PASN
    "MBEPASNNeuron",
    "build_mbe_pasn",
    # MBE-PASN-S
    "MBEPASNSNeuron",
    "build_mbe_pasn_s",
    # metrics
    "neuron_cost",
    "spikes_per_input",
]
