import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt

class QuantumForgeEnv(gym.Env):
    def __init__(self, num_qubits=3, backend='qiskit', noise_level=0.01):
        super(QuantumForgeEnv, self).__init__()
        self.num_qubits = num_qubits
        self.backend = backend.lower()
        self.noise_level = noise_level
        
        if self.backend not in ['qiskit', 'cirq']:
            raise ValueError("Backend must be either 'qiskit' or 'cirq'")
        
        self.circuit = None
        self.target_state = None
        
        # Action space: (operation, qubit1, qubit2, parameter)
        self.action_space = spaces.Box(
            low=np.array([0, 0, 0, -np.pi]),
            high=np.array([5, num_qubits-1, num_qubits-1, np.pi]),
            dtype=np.float32
        )
        
        # Observation space: complex amplitudes of the quantum state
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(2**num_qubits * 2,), dtype=np.float32
        )
        
        if self.backend == 'qiskit':
            from qiskit import QuantumCircuit
            from qiskit.quantum_info import state_fidelity, random_statevector
            from qiskit import Aer, execute
            from qiskit.providers.aer.noise import NoiseModel
            from qiskit.providers.aer.noise.errors import depolarizing_error
            self.QuantumCircuit = QuantumCircuit
            self.state_fidelity = state_fidelity
            self.random_statevector = random_statevector
            self.backend_simulator = Aer.get_backend('qasm_simulator')
            self.execute = execute
            self.noise_model = NoiseModel()
            self.noise_model.add_all_qubit_quantum_error(depolarizing_error(self.noise_level, 1), ['u1', 'u2', 'u3'])
        elif self.backend == 'cirq':
            import cirq
            self.QuantumCircuit = cirq.Circuit
            self.state_fidelity = cirq.fidelity
            self.random_statevector = lambda dim: cirq.testing.random_state_vector(dim, random_state=np.random).astype(np.complex128)
            self.backend_simulator = cirq.DensityMatrixSimulator(noise=cirq.depolarize(p=self.noise_level))

    def reset(self, seed=None):
        super().reset(seed=seed)
        if self.backend == 'qiskit':
            self.circuit = self.QuantumCircuit(self.num_qubits)
        elif self.backend == 'cirq':
            self.circuit = self.QuantumCircuit()
        self.target_state = self.random_statevector(2**self.num_qubits)
        return self._get_observation(), {}

    def step(self, action):
        operation, qubit1, qubit2, parameter = action
        operation = int(operation)
        qubit1 = int(qubit1)
        qubit2 = int(qubit2)

        if self.backend == 'qiskit':
            if operation == 0:
                self.circuit.x(qubit1)
            elif operation == 1:
                self.circuit.z(qubit1)
            elif operation == 2:
                self.circuit.h(qubit1)
            elif operation == 3:
                self.circuit.ry(parameter, qubit1)
            elif operation == 4:
                self.circuit.cx(qubit1, qubit2)
            elif operation == 5:
                self.circuit.cz(qubit1, qubit2)
        elif self.backend == 'cirq':
            q1 = cirq.LineQubit(qubit1)
            q2 = cirq.LineQubit(qubit2)
            if operation == 0:
                self.circuit.append(cirq.X(q1))
            elif operation == 1:
                self.circuit.append(cirq.Z(q1))
            elif operation == 2:
                self.circuit.append(cirq.H(q1))
            elif operation == 3:
                self.circuit.append(cirq.ry(parameter).on(q1))
            elif operation == 4:
                self.circuit.append(cirq.CNOT(q1, q2))
            elif operation == 5:
                self.circuit.append(cirq.CZ(q1, q2))

        obs = self._get_observation()
        reward = self._calculate_reward()
        done = (len(self.circuit) >= 20)  # Max circuit depth of 20
        return obs, reward, done, False, {}

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
        return np.concatenate([statevector.real, statevector.imag]).astype(np.float32)

    def _calculate_reward(self):
        current_state = self._get_observation()[:2**self.num_qubits] + 1j * self._get_observation()[2**self.num_qubits:]
        if self.backend == 'qiskit':
            fidelity = self.state_fidelity(current_state, self.target_state)
        elif self.backend == 'cirq':
            fidelity = self.state_fidelity(current_state, self.target_state)
        
        # Penalize for circuit depth
        depth_penalty = 0.01 * len(self.circuit)
        
        return fidelity - depth_penalty

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