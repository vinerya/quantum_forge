import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class HardwareConstraints:
    connectivity_map: Dict[int, List[int]]  # Qubit connectivity
    gate_times: Dict[str, float]            # Gate execution times
    error_rates: Dict[Tuple[int, int], float]  # Two-qubit gate error rates
    max_parallel_gates: int                 # Maximum parallel gate operations

class QuantumForgeEnv(gym.Env):
    def __init__(self, num_qubits=3, backend='qiskit', noise_level=0.01,
                 hardware_constraints: Optional[HardwareConstraints] = None,
                 optimization_metric='weighted'):
        super(QuantumForgeEnv, self).__init__()
        self.num_qubits = num_qubits
        self.backend = backend.lower()
        self.noise_level = noise_level
        self.hardware_constraints = hardware_constraints
        self.optimization_metric = optimization_metric
        
        if self.backend not in ['qiskit', 'cirq']:
            raise ValueError("Backend must be either 'qiskit' or 'cirq'")
        
        self.circuit = None
        self.target_state = None
        self.parallel_layers = []  # Track parallel gate operations
        
        # Extended action space with additional quantum operations
        # (operation, qubit1, qubit2, parameter1, parameter2)
        self.action_space = spaces.Box(
            low=np.array([0, 0, 0, -np.pi, -np.pi]),
            high=np.array([9, num_qubits-1, num_qubits-1, np.pi, np.pi]),
            dtype=np.float32
        )
        
        # Enhanced observation space including hardware state
        obs_dim = 2**num_qubits * 2  # Complex amplitudes
        if hardware_constraints:
            obs_dim += len(hardware_constraints.connectivity_map)  # Add connectivity state
        
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(obs_dim,), dtype=np.float32
        )
        
        self._setup_backend()
        self._initialize_cost_metrics()

    def _setup_backend(self):
        if self.backend == 'qiskit':
            from qiskit import QuantumCircuit
            from qiskit.quantum_info import state_fidelity, random_statevector
            from qiskit import Aer, execute
            from qiskit.providers.aer.noise import NoiseModel
            from qiskit.providers.aer.noise.errors import depolarizing_error, thermal_relaxation_error
            
            self.QuantumCircuit = QuantumCircuit
            self.state_fidelity = state_fidelity
            self.random_statevector = random_statevector
            self.backend_simulator = Aer.get_backend('qasm_simulator')
            self.execute = execute
            
            # Enhanced noise model
            self.noise_model = NoiseModel()
            self.noise_model.add_all_qubit_quantum_error(
                depolarizing_error(self.noise_level, 1), ['u1', 'u2', 'u3']
            )
            # Add T1/T2 relaxation noise
            self.noise_model.add_all_qubit_quantum_error(
                thermal_relaxation_error(t1=50.0, t2=70.0, time=0.1),
                ['u1', 'u2', 'u3', 'cx']
            )
            
        elif self.backend == 'cirq':
            import cirq
            self.QuantumCircuit = cirq.Circuit
            self.state_fidelity = cirq.fidelity
            self.random_statevector = lambda dim: cirq.testing.random_state_vector(dim, random_state=np.random).astype(np.complex128)
            
            # Enhanced noise simulation
            noise_ops = [
                cirq.depolarize(p=self.noise_level),
                cirq.amplitude_damp(gamma=0.1),
                cirq.phase_damp(gamma=0.05)
            ]
            self.backend_simulator = cirq.DensityMatrixSimulator(noise=noise_ops)

    def _initialize_cost_metrics(self):
        self.cost_metrics = {
            'gate_count': 1.0,
            'circuit_depth': 0.5,
            'two_qubit_gate_count': 2.0,
            'hardware_compliance': 1.5,
            'execution_time': 1.0
        }

    def reset(self, seed=None):
        super().reset(seed=seed)
        if self.backend == 'qiskit':
            self.circuit = self.QuantumCircuit(self.num_qubits)
        elif self.backend == 'cirq':
            self.circuit = self.QuantumCircuit()
        self.target_state = self.random_statevector(2**self.num_qubits)
        self.parallel_layers = []
        return self._get_observation(), {}

    def step(self, action):
        operation, qubit1, qubit2, param1, param2 = action
        operation = int(operation)
        qubit1 = int(qubit1)
        qubit2 = int(qubit2)

        # Check hardware constraints
        if self.hardware_constraints and not self._check_hardware_constraints(qubit1, qubit2, operation):
            return self._get_observation(), -1.0, True, False, {'error': 'Hardware constraints violated'}

        if self.backend == 'qiskit':
            self._apply_qiskit_operation(operation, qubit1, qubit2, param1, param2)
        elif self.backend == 'cirq':
            self._apply_cirq_operation(operation, qubit1, qubit2, param1, param2)

        obs = self._get_observation()
        reward = self._calculate_reward()
        done = self._check_termination()
        
        return obs, reward, done, False, self._get_info()

    def _apply_qiskit_operation(self, operation, qubit1, qubit2, param1, param2):
        # Extended quantum operations
        if operation == 0:
            self.circuit.x(qubit1)
        elif operation == 1:
            self.circuit.z(qubit1)
        elif operation == 2:
            self.circuit.h(qubit1)
        elif operation == 3:
            self.circuit.ry(param1, qubit1)
        elif operation == 4:
            self.circuit.cx(qubit1, qubit2)
        elif operation == 5:
            self.circuit.cz(qubit1, qubit2)
        elif operation == 6:
            self.circuit.rxx(param1, qubit1, qubit2)  # Ising XX coupling
        elif operation == 7:
            self.circuit.rzz(param1, qubit1, qubit2)  # Ising ZZ coupling
        elif operation == 8:
            self.circuit.cp(param1, qubit1, qubit2)   # Controlled phase rotation
        elif operation == 9:
            # Arbitrary single-qubit rotation (U3 gate)
            self.circuit.u(param1, param2, 0, qubit1)

    def _apply_cirq_operation(self, operation, qubit1, qubit2, param1, param2):
        q1 = cirq.LineQubit(qubit1)
        q2 = cirq.LineQubit(qubit2)
        
        # Extended quantum operations for Cirq
        ops = {
            0: cirq.X(q1),
            1: cirq.Z(q1),
            2: cirq.H(q1),
            3: cirq.ry(param1).on(q1),
            4: cirq.CNOT(q1, q2),
            5: cirq.CZ(q1, q2),
            6: cirq.XXPowGate(exponent=param1/np.pi)(q1, q2),
            7: cirq.ZZPowGate(exponent=param1/np.pi)(q1, q2),
            8: cirq.CZPowGate(exponent=param1/np.pi)(q1, q2),
            9: cirq.PhasedXPowGate(
                phase_exponent=param2/np.pi,
                exponent=param1/np.pi
            )(q1)
        }
        
        if operation in ops:
            self.circuit.append(ops[operation])

    def _check_hardware_constraints(self, qubit1, qubit2, operation) -> bool:
        if not self.hardware_constraints:
            return True
            
        # Check connectivity
        if operation in [4, 5, 6, 7, 8]:  # Two-qubit gates
            if qubit2 not in self.hardware_constraints.connectivity_map.get(qubit1, []):
                return False
                
        # Check parallel gate limit
        if len(self.parallel_layers[-1]) if self.parallel_layers else 0 >= self.hardware_constraints.max_parallel_gates:
            return False
            
        return True

    def _calculate_reward(self):
        current_state = self._get_observation()[:2**self.num_qubits] + 1j * self._get_observation()[2**self.num_qubits:]
        
        # State fidelity
        if self.backend == 'qiskit':
            fidelity = self.state_fidelity(current_state, self.target_state)
        elif self.backend == 'cirq':
            fidelity = self.state_fidelity(current_state, self.target_state)
        
        # Calculate costs based on optimization metric
        costs = 0
        if self.optimization_metric == 'weighted':
            costs = (
                self.cost_metrics['gate_count'] * len(self.circuit) +
                self.cost_metrics['circuit_depth'] * self._calculate_depth() +
                self.cost_metrics['two_qubit_gate_count'] * self._count_two_qubit_gates() +
                self.cost_metrics['hardware_compliance'] * self._calculate_hardware_compliance() +
                self.cost_metrics['execution_time'] * self._estimate_execution_time()
            )
        
        return fidelity - 0.01 * costs

    def _calculate_depth(self) -> int:
        if self.backend == 'qiskit':
            return self.circuit.depth()
        return len(self.parallel_layers)

    def _count_two_qubit_gates(self) -> int:
        count = 0
        if self.backend == 'qiskit':
            for instruction in self.circuit.data:
                if len(instruction.qubits) == 2:
                    count += 1
        else:
            for moment in self.circuit.moments:
                count += sum(1 for op in moment if len(op.qubits) == 2)
        return count

    def _calculate_hardware_compliance(self) -> float:
        if not self.hardware_constraints:
            return 0.0
            
        violations = 0
        if self.backend == 'qiskit':
            for instruction in self.circuit.data:
                if len(instruction.qubits) == 2:
                    q1, q2 = instruction.qubits
                    if q2.index not in self.hardware_constraints.connectivity_map.get(q1.index, []):
                        violations += 1
        return violations

    def _estimate_execution_time(self) -> float:
        if not self.hardware_constraints or not self.hardware_constraints.gate_times:
            return 0.0
            
        total_time = 0.0
        if self.backend == 'qiskit':
            for instruction in self.circuit.data:
                gate_name = instruction.operation.name
                total_time += self.hardware_constraints.gate_times.get(gate_name, 0.0)
        return total_time

    def _get_observation(self):
        if self.backend == 'qiskit':
            job = self.execute(self.circuit, self.backend_simulator, noise_model=self.noise_model, shots=1000)
            counts = job.result().get_counts()
            probabilities = np.zeros(2**self.num_qubits)
            for bitstring, count in counts.items():
                index = int(bitstring, 2)
                probabilities[index] = count / 1000
            statevector = np.sqrt(probabilities)
        elif self.backend == 'cirq':
            result = self.backend_simulator.simulate(self.circuit)
            statevector = result.final_state_vector
            
        obs = np.concatenate([statevector.real, statevector.imag]).astype(np.float32)
        
        # Add hardware state if constraints are present
        if self.hardware_constraints:
            hardware_state = self._get_hardware_state()
            obs = np.concatenate([obs, hardware_state])
            
        return obs

    def _get_hardware_state(self) -> np.ndarray:
        if not self.hardware_constraints:
            return np.array([])
            
        # Create a binary vector representing qubit connectivity availability
        state = []
        for q1 in range(self.num_qubits):
            for q2 in self.hardware_constraints.connectivity_map.get(q1, []):
                state.append(1.0 if q2 < self.num_qubits else 0.0)
                
        return np.array(state, dtype=np.float32)

    def _get_info(self) -> dict:
        return {
            'circuit_depth': self._calculate_depth(),
            'two_qubit_gate_count': self._count_two_qubit_gates(),
            'hardware_violations': self._calculate_hardware_compliance(),
            'estimated_execution_time': self._estimate_execution_time()
        }

    def _check_termination(self) -> bool:
        # Enhanced termination conditions
        if len(self.circuit) >= 30:  # Maximum circuit length
            return True
        if self._calculate_hardware_compliance() > 5:  # Too many hardware violations
            return True
        if self._estimate_execution_time() > 1000:  # Time limit exceeded
            return True
        return False

    def render(self, mode='human'):
        if mode == 'human':
            print(self.circuit)
        elif mode == 'rgb_array':
            return self._render_state()

    def _render_state(self):
        state = self._get_observation()[:2**self.num_qubits]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(range(len(state)), state)
        ax.set_xlabel('Basis State')
        ax.set_ylabel('Amplitude')
        ax.set_title('Quantum State Visualization')
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return img

    def close(self):
        pass
