"""Smoke-test the environment with a random agent. Needs only the core install."""

import gymnasium as gym
import numpy as np

import quantum_forge  # noqa: F401  (registers QuantumForge-v0)


def main() -> None:
    env = gym.make("QuantumForge-v0", num_qubits=3, target="ghz", render_mode="human")
    obs, info = env.reset(seed=0)
    total = 0.0
    done = False
    while not done:
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        total += reward
        done = terminated or truncated
    env.render()
    print(f"return={total:.3f} fidelity={info['fidelity']:.3f} gates={info['gate_count']} depth={info['circuit_depth']}")

    # A hand-written GHZ preparation reaches fidelity 1 in three steps.
    env = gym.make("QuantumForge-v0", num_qubits=3, target="ghz")
    env.reset(seed=0)
    for action in ([2, 0, 0, 0, 0], [4, 0, 1, 0, 0], [4, 1, 2, 0, 0]):
        obs, reward, terminated, truncated, info = env.step(np.array(action, dtype=np.float32))
    print(f"hand-written GHZ: fidelity={info['fidelity']:.3f} terminated={terminated}")


if __name__ == "__main__":
    main()
