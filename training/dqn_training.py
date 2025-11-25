"""
DQN Training Script for PharmaStock Environment
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from environment.custom_env import PharmaStockEnv
import json
import numpy as np
import time


def make_env():
    return PharmaStockEnv()


def train_dqn_config(config, config_name, save_dir):
    """Train DQN with specific configuration"""
    
    print(f"\n{'='*70}")
    print(f"Training DQN - {config_name}")
    print(f"{'='*70}")
    
    env = DummyVecEnv([make_env])
    eval_env = DummyVecEnv([make_env])
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=save_dir,
        eval_freq=5000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )
    
    model = DQN(
        'MlpPolicy',
        env,
        learning_rate=config['learning_rate'],
        gamma=config['gamma'],
        buffer_size=config['buffer_size'],
        batch_size=config['batch_size'],
        exploration_fraction=config['exploration_fraction'],
        exploration_initial_eps=config['exploration_initial_eps'],
        exploration_final_eps=config['exploration_final_eps'],
        target_update_interval=config['target_update_interval'],
        learning_starts=1000,
        train_freq=4,
        verbose=1,
        tensorboard_log=os.path.join(save_dir, 'tensorboard')
    )
    
    print(f"\nStarting training for {config['timesteps']:,} timesteps...")
    start_time = time.time()
    
    model.learn(total_timesteps=config['timesteps'], callback=[eval_callback], progress_bar=True)
    
    training_time = time.time() - start_time
    
    final_model_path = os.path.join(save_dir, f'{config_name}_final.zip')
    model.save(final_model_path)
    
    print(f"\n Training complete! Time: {training_time/60:.1f} minutes")
    
    # Evaluate
    rewards = []
    for _ in range(10):
        obs = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = env.step(action)
            episode_reward += reward[0]
            if dones[0]:
                rewards.append(episode_reward)
                done = True
    
    results = {
        'config_name': config_name,
        'training_time_minutes': training_time / 60,
        'mean_reward': float(np.mean(rewards)),
        'std_reward': float(np.std(rewards))
    }
    
    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    env.close()
    eval_env.close()
    return results


def main():
    print("\n" + "="*70)
    print("DQN TRAINING FOR PHARMASTOCK ENVIRONMENT")
    print("="*70)
    
    base_dir = 'models/dqn'
    os.makedirs(base_dir, exist_ok=True)
    
    configs = [
        {'name': 'config_01_baseline', 'learning_rate': 1e-4, 'gamma': 0.99, 'buffer_size': 50000, 'batch_size': 32, 'exploration_fraction': 0.1, 'exploration_initial_eps': 1.0, 'exploration_final_eps': 0.05, 'target_update_interval': 1000, 'timesteps': 100000},
        {'name': 'config_02_high_lr', 'learning_rate': 5e-4, 'gamma': 0.99, 'buffer_size': 50000, 'batch_size': 32, 'exploration_fraction': 0.1, 'exploration_initial_eps': 1.0, 'exploration_final_eps': 0.05, 'target_update_interval': 1000, 'timesteps': 80000},
        {'name': 'config_03_low_lr', 'learning_rate': 5e-5, 'gamma': 0.99, 'buffer_size': 50000, 'batch_size': 32, 'exploration_fraction': 0.1, 'exploration_initial_eps': 1.0, 'exploration_final_eps': 0.05, 'target_update_interval': 1000, 'timesteps': 100000},
        {'name': 'config_04_high_gamma', 'learning_rate': 1e-4, 'gamma': 0.995, 'buffer_size': 50000, 'batch_size': 32, 'exploration_fraction': 0.1, 'exploration_initial_eps': 1.0, 'exploration_final_eps': 0.05, 'target_update_interval': 1000, 'timesteps': 100000},
        {'name': 'config_05_large_batch', 'learning_rate': 1e-4, 'gamma': 0.99, 'buffer_size': 50000, 'batch_size': 64, 'exploration_fraction': 0.1, 'exploration_initial_eps': 1.0, 'exploration_final_eps': 0.05, 'target_update_interval': 1000, 'timesteps': 80000},
        {'name': 'config_06_small_batch', 'learning_rate': 1e-4, 'gamma': 0.99, 'buffer_size': 50000, 'batch_size': 16, 'exploration_fraction': 0.1, 'exploration_initial_eps': 1.0, 'exploration_final_eps': 0.05, 'target_update_interval': 1000, 'timesteps': 100000},
        {'name': 'config_07_high_exploration', 'learning_rate': 1e-4, 'gamma': 0.99, 'buffer_size': 50000, 'batch_size': 32, 'exploration_fraction': 0.3, 'exploration_initial_eps': 1.0, 'exploration_final_eps': 0.1, 'target_update_interval': 1000, 'timesteps': 80000},
        {'name': 'config_08_large_buffer', 'learning_rate': 1e-4, 'gamma': 0.99, 'buffer_size': 100000, 'batch_size': 32, 'exploration_fraction': 0.1, 'exploration_initial_eps': 1.0, 'exploration_final_eps': 0.05, 'target_update_interval': 1000, 'timesteps': 100000},
        {'name': 'config_09_fast_target', 'learning_rate': 1e-4, 'gamma': 0.99, 'buffer_size': 50000, 'batch_size': 32, 'exploration_fraction': 0.1, 'exploration_initial_eps': 1.0, 'exploration_final_eps': 0.05, 'target_update_interval': 500, 'timesteps': 80000},
        {'name': 'config_10_BEST', 'learning_rate': 1e-4, 'gamma': 0.995, 'buffer_size': 100000, 'batch_size': 64, 'exploration_fraction': 0.15, 'exploration_initial_eps': 1.0, 'exploration_final_eps': 0.02, 'target_update_interval': 1000, 'timesteps': 1000000}
    ]
    
    all_results = []
    for i, config in enumerate(configs, 1):
        print(f"\n{'#'*70}")
        print(f"CONFIGURATION {i}/10: {config['name']}")
        print(f"{'#'*70}")
        
        config_dir = os.path.join(base_dir, config['name'])
        os.makedirs(config_dir, exist_ok=True)
        
        try:
            results = train_dqn_config(config, config['name'], config_dir)
            all_results.append(results)
        except Exception as e:
            print(f"Error: {e}")
    
    with open(os.path.join(base_dir, 'training_summary.json'), 'w') as f:
        json.dump(all_results, f, indent=4)
    
    print("\nDQN Training Complete!")


if __name__ == "__main__":
    main()
