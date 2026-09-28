"""Train a PPO agent to prepare GHZ states with QuantumForge.

Requires the ``rl`` extra: ``pip install quantum-forge[rl,viz]``.
"""

import argparse

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

import quantum_forge  # noqa: F401  (registers QuantumForge-v0)


def run(backend: str, timesteps: int, num_qubits: int) -> None:
    print(f"\nTraining with the {backend} backend")
    env_kwargs = {"backend": backend, "num_qubits": num_qubits, "target": "ghz", "max_steps": 12}
    env = make_vec_env("QuantumForge-v0", n_envs=4, env_kwargs=env_kwargs)

    model = PPO("MlpPolicy", env, verbose=0, learning_rate=3e-4, n_steps=512, batch_size=64, ent_coef=0.01)
    model.learn(total_timesteps=timesteps)

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean episode reward: {mean_reward:.3f} +/- {std_reward:.3f}")

    # Roll out one greedy episode on a single environment and show the circuit.
    single = gym.make("QuantumForge-v0", render_mode="human", **env_kwargs)
    obs, info = single.reset(seed=0)
    rewards = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = single.step(action)
        rewards.append(reward)
        done = terminated or truncated
    print(f"Final fidelity: {info['fidelity']:.3f} with {info['gate_count']} gates")
    single.render()

    plt.figure(figsize=(8, 4))
    plt.plot(rewards)
    plt.title(f"Per-step reward ({backend})")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.savefig(f"rewards_{backend}.png")
    plt.close()
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="qiskit", choices=["qiskit", "cirq"])
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument("--qubits", type=int, default=3)
    args = parser.parse_args()
    run(args.backend, args.timesteps, args.qubits)
