# QuantumForge: Adaptive Quantum Circuit Synthesis

QuantumForge is a sophisticated quantum circuit design environment that focuses on adaptive synthesis of quantum circuits. It leverages reinforcement learning techniques to construct and optimize quantum circuits that generate desired quantum states.

## Features

- Support for both Qiskit and Cirq backends
- Gymnasium-compatible environment
- Advanced action space including multi-qubit gates
- Realistic noise simulation
- Reinforcement learning integration with Stable Baselines3
- Hyperparameter optimization using Optuna
- Circuit optimization and visualization

## Installation

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Basic Example

See `example.py` for a basic usage example. This script demonstrates:

- Training a PPO agent on the QuantumForge environment
- Evaluating the trained agent
- Visualizing rewards and final quantum states

To run the basic example:

```
python example.py
```

### Advanced Example

See `advanced_example.py` for a more sophisticated usage. This script showcases:

- Hyperparameter optimization using Optuna
- Training with optimized hyperparameters
- Circuit optimization
- Visualizing the optimized circuit

To run the advanced example:

```
python advanced_example.py
```

## Components

### Environment (QuantumForgeEnv)

- Observations: The current quantum state represented as a complex vector
- Actions: (operation, qubit1, qubit2, parameter)
  - Operations: X, Z, H, RY, CNOT, CZ
  - Qubits: Indices of qubits to apply operations
  - Parameter: Used for parameterized gates (e.g., RY)
- Reward: Based on the fidelity between the current state and the target state, with a penalty for circuit depth

### Reinforcement Learning

The project uses Stable Baselines3 for reinforcement learning algorithms. The examples demonstrate the use of PPO (Proximal Policy Optimization), but other algorithms like A2C and SAC are also available.

### Hyperparameter Optimization

Optuna is used for hyperparameter optimization in the advanced example. This allows for automatic tuning of the RL algorithm's hyperparameters to achieve better performance.

## Backends

QuantumForge supports two quantum computing backends:

1. Qiskit: IBM's open-source framework for quantum computing
2. Cirq: Google's framework for writing, manipulating, and optimizing quantum circuits

Both backends provide similar functionality for the purposes of this environment, including noise simulation.

## Visualization

The examples generate various visualizations:

- Reward plots over time
- Final quantum state visualizations
- Optimized circuit visualization

These are saved as PNG files in the project directory.

## Extending the Project

You can extend this project in several ways:

1. Implement more quantum gates in the environment
2. Add support for more backends (e.g., PennyLane, QuTiP)
3. Implement more sophisticated reward functions
4. Explore different RL algorithms and architectures
5. Add more circuit optimization techniques

QuantumForge serves as a powerful platform for exploring the intersection of quantum computing and machine learning, offering a flexible environment for developing and optimizing quantum circuits using cutting-edge adaptive techniques.