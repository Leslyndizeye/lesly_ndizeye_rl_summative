"""
REINFORCE (Monte Carlo Policy Gradient) Training Script
Enhanced with Variance Reduction (Baseline) and Entropy Regularization
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from environment.custom_env import PharmaStockEnv
import json
import time


class PolicyNetwork(nn.Module):
    """Policy network for REINFORCE"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)


class REINFORCE:
    """REINFORCE Algorithm with Entropy and Baseline"""
    
    def __init__(self, state_dim, action_dim, learning_rate=1e-3, gamma=0.99, entropy_coef=0.01):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.entropy_coef = entropy_coef  # Encourages exploration
        
        self.saved_log_probs = []
        self.rewards = []
        self.entropies = [] # Track entropy for logging
    
    def select_action(self, state, deterministic=False):
        """Select action using policy network"""
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy(state)
        
        if deterministic:
            action = torch.argmax(probs, dim=1).item()
        else:
            m = Categorical(probs)
            action = m.sample()
            self.saved_log_probs.append(m.log_prob(action))
            self.entropies.append(m.entropy()) # Save entropy
            action = action.item()
        
        return action
    
    def update(self):
        """Update policy using REINFORCE with Baseline"""
        R = 0
        policy_loss = []
        returns = []
        
        # Calculate discounted returns
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns)
        
        # Normalize returns (Simple Baseline)
        # This significantly reduces variance
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        # Calculate loss with Entropy Regularization
        # Loss = -log(prob) * return - (entropy_coef * entropy)
        for log_prob, R, entropy in zip(self.saved_log_probs, returns, self.entropies):
            loss = -log_prob * R - (self.entropy_coef * entropy)
            policy_loss.append(loss)
        
        # Update policy
        self.optimizer.zero_grad()
        # Sum losses and perform backward pass
        loss_sum = torch.stack(policy_loss).sum()
        loss_sum.backward()
        
        # Gradient clipping prevents exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        
        self.optimizer.step()
        
        # Clear episode memory
        del self.rewards[:]
        del self.saved_log_probs[:]
        del self.entropies[:]
        
        return loss_sum.item()
    
    def save(self, path):
        """Save model"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
    
    def load(self, path):
        """Load model"""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def train_reinforce_config(config, config_name, save_dir):
    """Train REINFORCE with specific configuration"""
    
    print(f"\n{'='*70}")
    print(f"Training REINFORCE - {config_name}")
    print(f"{'='*70}")
    
    env = PharmaStockEnv()
    
    # Extract config parameters with defaults
    lr = config.get('learning_rate', 1e-3)
    gamma = config.get('gamma', 0.99)
    entropy_coef = config.get('entropy_coef', 0.01)
    num_episodes = config.get('num_episodes', 1000)
    
    agent = REINFORCE(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        learning_rate=lr,
        gamma=gamma,
        entropy_coef=entropy_coef
    )
    
    print(f"\nTraining for {num_episodes} episodes...")
    
    episode_rewards = []
    episode_lengths = []
    training_losses = []
    best_reward = -float('inf')
    start_time = time.time()
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done and episode_length < 1000:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            agent.rewards.append(reward)
            episode_reward += reward
            episode_length += 1
            
            state = next_state
            done = terminated or truncated
        
        # Update policy after each episode
        if len(agent.saved_log_probs) > 0:
            loss = agent.update()
            training_losses.append(loss)
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            best_model_path = os.path.join(save_dir, 'best_model.pth')
            agent.save(best_model_path)
        
        # Logging
        if (episode + 1) % 50 == 0:
            recent_rewards = episode_rewards[-50:]
            avg_reward = np.mean(recent_rewards)
            print(f"Episode {episode+1}/{num_episodes} | Avg Reward: {avg_reward:.2f} | Best: {best_reward:.2f}")
    
    training_time = time.time() - start_time
    
    final_model_path = os.path.join(save_dir, f'{config_name}_final.pth')
    agent.save(final_model_path)
    
    print(f"\n Training complete! Time: {training_time/60:.1f} minutes")
    
    # Evaluate Final Model
    agent.load(best_model_path)
    eval_rewards = []
    eval_service_levels = []
    eval_stockouts = []
    
    print("\nEvaluating Best Model...")
    for _ in range(10):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 1000:
            action = agent.select_action(state, deterministic=True)
            state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            done = terminated or truncated
        
        eval_rewards.append(episode_reward)
        eval_service_levels.append(info.get('service_level', 0))
        eval_stockouts.append(info.get('stockouts', 0))
    
    results = {
        'config_name': config_name,
        'config': config,
        'training_time_minutes': training_time / 60,
        'mean_reward': float(np.mean(eval_rewards)),
        'std_reward': float(np.std(eval_rewards)),
        'mean_service_level': float(np.mean(eval_service_levels)),
        'mean_stockouts': float(np.mean(eval_stockouts)),
        'best_training_reward': float(best_reward),
        'training_losses': training_losses[-100:] if training_losses else [] # Save last 100 losses
    }
    
    results_path = os.path.join(save_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResults: Mean Reward: {results['mean_reward']:.2f}, Service Level: {results['mean_service_level']*100:.1f}%")
    
    return results


def main():
    print("\n" + "="*70)
    print("REINFORCE TRAINING FOR PHARMASTOCK ENVIRONMENT")
    print("="*70)
    
    base_dir = 'models/reinforce'
    os.makedirs(base_dir, exist_ok=True)
    
    # Configurations covering varied Hyperparameters for analysis
    configs = [
        {'name': 'config_01_baseline', 'learning_rate': 3e-4, 'gamma': 0.99, 'num_episodes': 4000, 'entropy_coef': 0.01},
        {'name': 'config_02_high_lr', 'learning_rate': 1e-3, 'gamma': 0.99, 'num_episodes': 4000, 'entropy_coef': 0.01},
        {'name': 'config_03_low_lr', 'learning_rate': 1e-4, 'gamma': 0.99, 'num_episodes': 4000, 'entropy_coef': 0.01},
        {'name': 'config_04_high_gamma', 'learning_rate': 3e-4, 'gamma': 0.995, 'num_episodes': 4000, 'entropy_coef': 0.01},
        {'name': 'config_05_low_gamma', 'learning_rate': 3e-4, 'gamma': 0.95, 'num_episodes': 4000, 'entropy_coef': 0.01},
        {'name': 'config_06_high_entropy', 'learning_rate': 3e-4, 'gamma': 0.99, 'num_episodes': 4000, 'entropy_coef': 0.05},
        {'name': 'config_07_no_entropy', 'learning_rate': 3e-4, 'gamma': 0.99, 'num_episodes': 4000, 'entropy_coef': 0.0},
        {'name': 'config_08_balanced', 'learning_rate': 3e-4, 'gamma': 0.98, 'num_episodes': 4000, 'entropy_coef': 0.01},
        {'name': 'config_09_long_train', 'learning_rate': 2e-4, 'gamma': 0.99, 'num_episodes': 4000, 'entropy_coef': 0.01},
        {'name': 'config_10_BEST', 'learning_rate': 3e-4, 'gamma': 0.99, 'num_episodes': 15000, 'entropy_coef': 0.01}
    ]
    
    all_results = []
    
    for i, config in enumerate(configs, 1):
        print(f"\n\n{'#'*70}")
        print(f"CONFIGURATION {i}/10: {config['name']}")
        print(f"{'#'*70}")
        
        config_dir = os.path.join(base_dir, config['name'])
        os.makedirs(config_dir, exist_ok=True)
        
        try:
            results = train_reinforce_config(config, config['name'], config_dir)
            all_results.append(results)
        except Exception as e:
            print(f"\n Error training {config['name']}: {e}")
            import traceback
            traceback.print_exc()
    
    summary_path = os.path.join(base_dir, 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    print("\n" + "="*70)
    print("REINFORCE TRAINING COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()