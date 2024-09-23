import gymnasium as gym
from stable_baselines3 import PPO, A2C, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
import optuna
import matplotlib.pyplot as plt
import numpy as np
from quantum_forge import QuantumForgeEnv

# Register the environment
gym.register(
    id='QuantumForge-v0',
    entry_point='quantum_forge:QuantumForgeEnv',
)

def optimize_circuit(model, env, n_steps=100):
    obs = env.reset()
    circuit = []
    for _ in range(n_steps):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        circuit.append(action)
        if done:
            break
    return circuit, reward

def objective(trial):
    # Hyperparameters to be optimized
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    n_steps = trial.suggest_categorical('n_steps', [1024, 2048, 4096])
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    n_epochs = trial.suggest_int('n_epochs', 5, 20)
    ent_coef = trial.suggest_loguniform('ent_coef', 1e-8, 1e-1)

    # Create environment
    env = make_vec_env('QuantumForge-v0', n_envs=4, env_kwargs={'backend': 'qiskit', 'num_qubits': 3, 'noise_level': 0.01})

    # Create model with trial hyperparameters
    model = PPO("MlpPolicy", env, verbose=0, learning_rate=lr, n_steps=n_steps, 
                batch_size=batch_size, n_epochs=n_epochs, ent_coef=ent_coef)

    # Train the agent
    eval_callback = EvalCallback(env, best_model_save_path='./logs/',
                                 log_path='./logs/', eval_freq=10000,
                                 deterministic=True, render=False)
    model.learn(total_timesteps=100000, callback=eval_callback)

    # Evaluate the agent
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
    
    return mean_reward

def run_advanced_simulation():
    print("\nRunning advanced simulation:")

    # Hyperparameter optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)

    print('Best trial:')
    trial = study.best_trial
    print(f'Value: {trial.value}')
    print('Params: ')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')

    # Create environment with best hyperparameters
    env = make_vec_env('QuantumForge-v0', n_envs=4, env_kwargs={'backend': 'qiskit', 'num_qubits': 3, 'noise_level': 0.01})

    # Create and train the agent with best hyperparameters
    model = PPO("MlpPolicy", env, verbose=1, **trial.params)
    model.learn(total_timesteps=200000)

    # Optimize circuit
    optimized_circuit, final_reward = optimize_circuit(model, env)

    print(f"Optimized circuit length: {len(optimized_circuit)}")
    print(f"Final reward: {final_reward}")

    # Visualize optimized circuit
    plt.figure(figsize=(12, 6))
    plt.imshow(np.array(optimized_circuit).T, aspect='auto', cmap='viridis')
    plt.title('Optimized Quantum Circuit')
    plt.xlabel('Time Step')
    plt.ylabel('Action Component')
    plt.colorbar(label='Action Value')
    plt.savefig('optimized_circuit.png')
    plt.close()

    env.close()

if __name__ == "__main__":
    run_advanced_simulation()
    print("\nAdvanced simulation completed. Check the generated PNG file for visualization.")