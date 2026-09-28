"""Train on a hardware-constrained environment with a heavy-hex style coupling map.

Requires the ``rl`` extra: ``pip install quantum-forge[rl]``.
"""

import argparse

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import quantum_forge  # noqa: F401  (registers QuantumForge-v0)
from quantum_forge import HardwareConstraints


def linear_chain(num_qubits: int) -> HardwareConstraints:
    """Nearest-neighbour chain with IBM-like gate times."""
    connectivity = {i: [] for i in range(num_qubits)}
    for i in range(num_qubits - 1):
        connectivity[i].append(i + 1)
        connectivity[i + 1].append(i)
    return HardwareConstraints(
        connectivity_map=connectivity,
        gate_times={"x": 50, "z": 0, "h": 50, "ry": 50, "u": 50, "cx": 300, "cz": 300, "rxx": 400, "rzz": 400, "cp": 350},
        error_rates={(i, i + 1): 0.01 for i in range(num_qubits - 1)},
        max_parallel_gates=num_qubits,
    )


def run(timesteps: int, num_qubits: int) -> None:
    hc = linear_chain(num_qubits)
    env_kwargs = {"num_qubits": num_qubits, "hardware_constraints": hc, "target": "ghz", "max_steps": 15}
    env = make_vec_env("QuantumForge-v0", n_envs=4, env_kwargs=env_kwargs)

    model = PPO("MlpPolicy", env, verbose=0, policy_kwargs={"net_arch": [128, 128]})
    model.learn(total_timesteps=timesteps)

    single = gym.make("QuantumForge-v0", render_mode="human", **env_kwargs)
    obs, info = single.reset(seed=0)
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = single.step(action)
        done = terminated or truncated

    print("\nFinal circuit:")
    single.render()
    print(f"Fidelity:               {info['fidelity']:.3f}")
    print(f"Depth:                  {info['circuit_depth']}")
    print(f"Two-qubit gates:        {info['two_qubit_gate_count']}")
    print(f"Hardware violations:    {info['hardware_violations']}")
    print(f"Estimated time (ns):    {info['estimated_execution_time']:.0f}")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument("--qubits", type=int, default=4)
    args = parser.parse_args()
    run(args.timesteps, args.qubits)
