import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import numpy as np
from quantum_forge import QuantumForgeEnv, HardwareConstraints

# Register the environment
gym.register(
    id='QuantumForge-v0',
    entry_point='quantum_forge:QuantumForgeEnv',
)

def create_ibmq_hardware_constraints():
    """Create hardware constraints based on IBMQ architecture"""
    return HardwareConstraints(
        # Toronto processor connectivity
        connectivity_map={
            0: [1, 2, 3],
            1: [0, 4],
            2: [0, 5],
            3: [0, 6],
            4: [1, 7],
            5: [2, 8],
            6: [3, 9],
            7: [4, 10],
            8: [5, 11],
            9: [6, 12],
            10: [7, 13],
            11: [8, 14],
            12: [9, 15],
            13: [10, 14],
            14: [11, 13],
            15: [12]
        },
        # Approximate gate times in ns
        gate_times={
            'x': 50,
            'sx': 50,
            'rz': 0,
            'cx': 300,
            'reset': 1000,
            'measure': 1000
        },
        # Two-qubit gate error rates
        error_rates={
            (0, 1): 0.01,
            (1, 4): 0.015,
            (2, 5): 0.012,
            # ... other error rates ...
        },
        max_parallel_gates=8
    )

def run_hardware_aware_optimization():
    print("\nRunning hardware-aware quantum circuit optimization:")
    
    # Create hardware constraints
    hardware_constraints = create_ibmq_hardware_constraints()
    
    # Create vectorized environment with hardware constraints
    env = make_vec_env(
        'QuantumForge-v0',
        n_envs=4,
        env_kwargs={
            'backend': 'qiskit',
            'num_qubits': 5,
            'noise_level': 0.01,
            'hardware_constraints': hardware_constraints,
            'optimization_metric': 'weighted'
        }
    )

    # Create and train the agent with hardware-aware parameters
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.01,
        policy_kwargs={
            "net_arch": [256, 256]  # Larger network for hardware-aware optimization
        }
    )

    # Add evaluation callback
    eval_callback = EvalCallback(
        env,
        best_model_save_path='./logs/',
        log_path='./logs/',
        eval_freq=1000,
        deterministic=True,
        render=False
    )

    # Train the agent
    model.learn(total_timesteps=100000, callback=eval_callback)

    # Evaluate the optimized circuit
    obs = env.reset()
    circuit_metrics = []
    
    for _ in range(100):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        
        # Collect metrics
        if not isinstance(infos, dict):
            infos = infos[0]  # Handle vectorized environment
        circuit_metrics.append({
            'depth': infos['circuit_depth'],
            'two_qubit_gates': infos['two_qubit_gate_count'],
            'hardware_violations': infos['hardware_violations'],
            'execution_time': infos['estimated_execution_time']
        })
        
        if dones.any():
            break

    # Print optimization results
    print("\nOptimization Results:")
    print("Final Circuit Metrics:")
    final_metrics = circuit_metrics[-1]
    print(f"Circuit Depth: {final_metrics['depth']}")
    print(f"Two-Qubit Gates: {final_metrics['two_qubit_gates']}")
    print(f"Hardware Violations: {final_metrics['hardware_violations']}")
    print(f"Estimated Execution Time: {final_metrics['execution_time']} ns")

    # Calculate improvement metrics
    initial_metrics = circuit_metrics[0]
    depth_reduction = (initial_metrics['depth'] - final_metrics['depth']) / initial_metrics['depth'] * 100
    gate_reduction = (initial_metrics['two_qubit_gates'] - final_metrics['two_qubit_gates']) / initial_metrics['two_qubit_gates'] * 100
    
    print(f"\nImprovements:")
    print(f"Depth Reduction: {depth_reduction:.2f}%")
    print(f"Two-Qubit Gate Reduction: {gate_reduction:.2f}%")

    env.close()

if __name__ == "__main__":
    run_hardware_aware_optimization()
    print("\nHardware-aware optimization completed. Check the logs directory for detailed results.")
