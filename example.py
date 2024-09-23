import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt
import numpy as np
from quantum_forge import QuantumForgeEnv

# Register the environment
gym.register(
    id='QuantumForge-v0',
    entry_point='quantum_forge:QuantumForgeEnv',
)

def run_simulation(backend):
    print(f"\nRunning simulation with {backend} backend:")
    
    # Create vectorized environment
    env = make_vec_env('QuantumForge-v0', n_envs=4, env_kwargs={'backend': backend, 'num_qubits': 3, 'noise_level': 0.01})

    # Create and train the agent
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99, ent_coef=0.01)
    model.learn(total_timesteps=50000)

    # Evaluate the agent
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Run a single episode for visualization
    obs = env.reset()
    rewards = []
    for _ in range(100):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        rewards.append(reward)
        if done.any():
            break

    # Plot rewards
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title(f'Rewards over time ({backend} backend)')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.savefig(f'rewards_{backend}.png')
    plt.close()

    # Visualize final quantum state
    final_state = env.render(mode='rgb_array')[0]
    plt.figure(figsize=(10, 5))
    plt.imshow(final_state)
    plt.title(f'Final Quantum State ({backend} backend)')
    plt.axis('off')
    plt.savefig(f'final_state_{backend}.png')
    plt.close()

    env.close()

# Run simulations with both backends
run_simulation('qiskit')
run_simulation('cirq')

print("\nSimulations completed. Check the generated PNG files for visualizations.")