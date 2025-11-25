"""
A2C Training Script for PharmaStock Environment
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from environment.custom_env import PharmaStockEnv
import json
import numpy as np
import time


def make_env():
    return PharmaStockEnv()


def train_a2c_config(config, config_name, save_dir):
    """Train A2C with specific configuration"""
    
    print(f"\n{'='*70}")
    print(f"Training A2C - {config_name}")
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
    
    model = A2C(
        'MlpPolicy',
        env,
        learning_rate=config['learning_rate'],
        gamma=config['gamma'],
        n_steps=config['n_steps'],
        ent_coef=config['ent_coef'],
        vf_coef=config['vf_coef'],
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
    print("A2C TRAINING FOR PHARMASTOCK ENVIRONMENT")
    print("="*70)
    
    base_dir = 'models/a2c'
    os.makedirs(base_dir, exist_ok=True)
    
    configs = [
        {'name': 'config_01_baseline', 'learning_rate': 7e-4, 'gamma': 0.99, 'n_steps': 5, 'ent_coef': 0.0, 'vf_coef': 0.5, 'timesteps': 100000},
        {'name': 'config_02_high_lr', 'learning_rate': 1e-3, 'gamma': 0.99, 'n_steps': 5, 'ent_coef': 0.0, 'vf_coef': 0.5, 'timesteps': 80000},
        {'name': 'config_03_low_lr', 'learning_rate': 3e-4, 'gamma': 0.99, 'n_steps': 5, 'ent_coef': 0.0, 'vf_coef': 0.5, 'timesteps': 100000},
        {'name': 'config_04_high_gamma', 'learning_rate': 7e-4, 'gamma': 0.995, 'n_steps': 5, 'ent_coef': 0.0, 'vf_coef': 0.5, 'timesteps': 100000},
        {'name': 'config_05_long_steps', 'learning_rate': 7e-4, 'gamma': 0.99, 'n_steps': 16, 'ent_coef': 0.0, 'vf_coef': 0.5, 'timesteps': 80000},
        {'name': 'config_06_entropy', 'learning_rate': 7e-4, 'gamma': 0.99, 'n_steps': 5, 'ent_coef': 0.01, 'vf_coef': 0.5, 'timesteps': 100000},
        {'name': 'config_07_high_vf', 'learning_rate': 7e-4, 'gamma': 0.99, 'n_steps': 5, 'ent_coef': 0.0, 'vf_coef': 0.75, 'timesteps': 80000},
        {'name': 'config_08_low_vf', 'learning_rate': 7e-4, 'gamma': 0.99, 'n_steps': 5, 'ent_coef': 0.0, 'vf_coef': 0.25, 'timesteps': 100000},
        {'name': 'config_09_balanced', 'learning_rate': 5e-4, 'gamma': 0.99, 'n_steps': 8, 'ent_coef': 0.005, 'vf_coef': 0.5, 'timesteps': 80000},
        {'name': 'config_10_BEST', 'learning_rate': 7e-4, 'gamma': 0.995, 'n_steps': 8, 'ent_coef': 0.005, 'vf_coef': 0.5, 'timesteps': 1000000}
    ]
    
    all_results = []
    for i, config in enumerate(configs, 1):
        print(f"\n{'#'*70}")
        print(f"CONFIGURATION {i}/10: {config['name']}")
        print(f"{'#'*70}")
        
        config_dir = os.path.join(base_dir, config['name'])
        os.makedirs(config_dir, exist_ok=True)
        
        try:
            results = train_a2c_config(config, config['name'], config_dir)
            all_results.append(results)
        except Exception as e:
            print(f"Error: {e}")
    
    with open(os.path.join(base_dir, 'training_summary.json'), 'w') as f:
        json.dump(all_results, f, indent=4)
    
    print("\nA2C Training Complete!")


if __name__ == "__main__":
    main()
