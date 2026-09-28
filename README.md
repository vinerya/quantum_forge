# QuantumForge

A [Gymnasium](https://gymnasium.farama.org/) environment for learning
hardware-aware quantum circuit synthesis.

An agent builds a circuit one gate at a time. Each step it picks an operation,
the qubits to act on and rotation angles. The reward is the fidelity between the
circuit's output state and a target state, minus a small weighted cost for gate
count, depth, two-qubit gates, hardware violations and estimated execution
time. Two simulation backends are supported, Qiskit and Cirq, with identical
semantics.

## Install

```bash
pip install -e .                 # core: gymnasium, numpy, qiskit, networkx
pip install -e ".[aer]"          # + noisy simulation via qiskit-aer
pip install -e ".[cirq]"         # + cirq backend
pip install -e ".[rl,viz]"       # + stable-baselines3, torch, optuna, matplotlib
pip install -e ".[dev]"          # everything needed to run the tests
```

## Quick start

```python
import gymnasium as gym
import numpy as np
import quantum_forge  # registers QuantumForge-v0

env = gym.make("QuantumForge-v0", num_qubits=3, target="ghz", render_mode="human")
obs, info = env.reset(seed=0)

# action = (operation, qubit1, qubit2, param1, param2); see quantum_forge.OPERATIONS
for action in ([2, 0, 0, 0, 0],   # H on qubit 0
               [4, 0, 1, 0, 0],   # CX 0 -> 1
               [4, 1, 2, 0, 0]):  # CX 1 -> 2
    obs, reward, terminated, truncated, info = env.step(np.array(action, dtype=np.float32))

env.render()
print(info["fidelity"], terminated)   # 1.0 True
```

Run a smoke test with a random agent (core install only):

```bash
python examples/random_agent.py
```

Train a PPO agent to prepare GHZ states (`rl` extra):

```bash
python examples/train_ppo.py --backend qiskit --timesteps 20000
```

## Environment

| | |
|---|---|
| **Action** | `Box(5)`: `(operation, qubit1, qubit2, param1, param2)`. Continuous values are rounded and clipped, so any continuous-control agent works. |
| **Operations** | `x z h ry cx cz rxx rzz cp u` (indices 0 to 9). Indices 4 to 8 are two-qubit gates. |
| **Observation** | Real and imaginary parts of the current statevector, plus one busy flag per qubit for the latest circuit layer when hardware constraints are set. |
| **Reward** | `fidelity - cost_scale * (w_gates * gates + w_depth * depth + w_2q * two_qubit_gates + w_viol * violations + w_time * exec_time)` |
| **Termination** | Fidelity reaches `success_threshold` (default 0.99), or the agent exceeds `max_violations` rejected hardware-violating actions. |
| **Truncation** | `max_steps` (default 30). |

### Constructor options

```python
QuantumForgeEnv(
    num_qubits=3,
    backend="qiskit",             # or "cirq"
    noise_level=0.0,              # per-gate depolarising probability; >0 needs qiskit-aer for qiskit
    hardware_constraints=None,    # HardwareConstraints(...)
    target="random",              # "random" (Haar, reseeded each reset), "ghz", or a statevector
    max_steps=30,
    success_threshold=0.99,
    max_violations=5,
    cost_weights=None,            # override any of the default weights
    cost_scale=0.01,
    render_mode=None,             # "human" prints the circuit, "rgb_array" returns a probability bar chart
)
```

### Hardware constraints

```python
from quantum_forge import HardwareConstraints

hc = HardwareConstraints(
    connectivity_map={0: [1], 1: [0, 2], 2: [1]},   # treated as undirected
    gate_times={"h": 50, "cx": 300},                # any units; summed into execution_time
    error_rates={(0, 1): 0.01, (1, 2): 0.015},
    max_parallel_gates=2,                           # caps gates per layer, which raises depth
)
env = QuantumForgeEnv(num_qubits=3, hardware_constraints=hc)
```

Two-qubit gates on non-adjacent qubits are rejected: the gate is not applied,
the reward is -1 and the violation count increases. Depth is computed with an
as-soon-as-possible layering that respects `max_parallel_gates`.

The underlying circuit is always available as `env.unwrapped.circuit`, a
`qiskit.QuantumCircuit` or `cirq.Circuit`, so it can be handed to any transpiler
or mitigation library.

## Circuit cutting

`QuantumCircuitCutter` slices a wide Qiskit circuit into consecutive segments
remapped onto only the qubits they touch, and reports which qubits carry state
across each cut.

```python
from quantum_forge import QuantumCircuitCutter

cutter = QuantumCircuitCutter(max_subcircuit_depth=50)
segments = cutter.cut_circuit(circuit)
```

`reconstruct_results` is a naive merge of per-segment counts, not a
quasi-probability reconstruction. See `examples/circuit_cutting.py`.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The suite covers both backends, verifies that every operation produces the same
state under Qiskit and Cirq, checks Gymnasium's `check_env`, and exercises
hardware constraints, noise, seeding and the cutter.

## Legacy code

`legacy/` holds earlier, unfinished drafts of a circuit optimizer, dynamic
compiler and error mitigator. They are not installed, not tested and do not
run. See `legacy/README.md`.

## License

MIT. See `LICENSE`.

## Citation

```bibtex
@software{quantumforge2024,
  title  = {QuantumForge: a Gymnasium environment for hardware-aware quantum circuit synthesis},
  year   = {2024},
  author = {Moudather Chelbi},
  url    = {https://github.com/vinerya/quantum_forge}
}
```
