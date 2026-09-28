"""QuantumForge: a Gymnasium environment for hardware-aware quantum circuit synthesis."""

from gymnasium.envs.registration import register

from .env import HardwareConstraints, QuantumForgeEnv, OPERATIONS
from .cutting import CircuitCut, QuantumCircuitCutter, SubCircuit

__all__ = [
    "HardwareConstraints",
    "QuantumForgeEnv",
    "OPERATIONS",
    "CircuitCut",
    "QuantumCircuitCutter",
    "SubCircuit",
]

__version__ = "0.2.0"

register(
    id="QuantumForge-v0",
    entry_point="quantum_forge.env:QuantumForgeEnv",
)
