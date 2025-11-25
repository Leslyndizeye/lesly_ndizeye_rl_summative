"""
PPO Training Script for PharmaStock Environment
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from environment.custom_env import PharmaStockEnv
import json
import numpy as np
import time


def make_env():
    return PharmaStockEnv()


def train_ppo_config(config, config_name, save_dir):
    """Train PPO with specific configuration"""
    
    print(f"\n{'='*70}")
    print(f"Training PPO - {config_name}")
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
    
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=config['learning_rate'],
        gamma=config['gamma'],
        n_steps=config['n_steps'],
        batch_size=config['batch_size'],
        n_epochs=config['n_epochs'],
        clip_range=config['clip_range'],
        ent_coef=config['ent_coef'],
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
    print("PPO TRAINING FOR PHARMASTOCK ENVIRONMENT")
    print("="*70)
    
    base_dir = 'models/ppo'
    os.makedirs(base_dir, exist_ok=True)
    
    configs = [
        {'name': 'config_01_baseline', 'learning_rate': 3e-4, 'gamma': 0.99, 'n_steps': 2048, 'batch_size': 64, 'n_epochs': 10, 'clip_range': 0.2, 'ent_coef': 0.0, 'timesteps': 100000},
        {'name': 'config_02_high_lr', 'learning_rate': 1e-3, 'gamma': 0.99, 'n_steps': 2048, 'batch_size': 64, 'n_epochs': 10, 'clip_range': 0.2, 'ent_coef': 0.0, 'timesteps': 80000},
        {'name': 'config_03_low_lr', 'learning_rate': 1e-4, 'gamma': 0.99, 'n_steps': 2048, 'batch_size': 64, 'n_epochs': 10, 'clip_range': 0.2, 'ent_coef': 0.0, 'timesteps': 100000},
        {'name': 'config_04_high_gamma', 'learning_rate': 3e-4, 'gamma': 0.995, 'n_steps': 2048, 'batch_size': 64, 'n_epochs': 10, 'clip_range': 0.2, 'ent_coef': 0.0, 'timesteps': 100000},
        {'name': 'config_05_large_batch', 'learning_rate': 3e-4, 'gamma': 0.99, 'n_steps': 2048, 'batch_size': 128, 'n_epochs': 10, 'clip_range': 0.2, 'ent_coef': 0.0, 'timesteps': 80000},
        {'name': 'config_06_more_epochs', 'learning_rate': 3e-4, 'gamma': 0.99, 'n_steps': 2048, 'batch_size': 64, 'n_epochs': 20, 'clip_range': 0.2, 'ent_coef': 0.0, 'timesteps': 100000},
        {'name': 'config_07_tight_clip', 'learning_rate': 3e-4, 'gamma': 0.99, 'n_steps': 2048, 'batch_size': 64, 'n_epochs': 10, 'clip_range': 0.1, 'ent_coef': 0.0, 'timesteps': 80000},
        {'name': 'config_08_entropy', 'learning_rate': 3e-4, 'gamma': 0.99, 'n_steps': 2048, 'batch_size': 64, 'n_epochs': 10, 'clip_range': 0.2, 'ent_coef': 0.01, 'timesteps': 100000},
        {'name': 'config_09_short_steps', 'learning_rate': 3e-4, 'gamma': 0.99, 'n_steps': 512, 'batch_size': 64, 'n_epochs': 10, 'clip_range': 0.2, 'ent_coef': 0.0, 'timesteps': 80000},
        {'name': 'config_10_BEST', 'learning_rate': 3e-4, 'gamma': 0.995, 'n_steps': 2048, 'batch_size': 64, 'n_epochs': 10, 'clip_range': 0.2, 'ent_coef': 0.005, 'timesteps': 1000000}
    ]
    
    all_results = []
    for i, config in enumerate(configs, 1):
        print(f"\n{'#'*70}")
        print(f"CONFIGURATION {i}/10: {config['name']}")
        print(f"{'#'*70}")
        
        config_dir = os.path.join(base_dir, config['name'])
        os.makedirs(config_dir, exist_ok=True)
        
        try:
            results = train_ppo_config(config, config['name'], config_dir)
            all_results.append(results)
        except Exception as e:
            print(f"Error: {e}")
    
    with open(os.path.join(base_dir, 'training_summary.json'), 'w') as f:
        json.dump(all_results, f, indent=4)
    
    print("\nPPO Training Complete!")


if __name__ == "__main__":
    main()
