"""
Main entry point for PharmaStock RL Project
Supports running trained models with visualization
"""

import argparse
import os
import time
from stable_baselines3 import PPO, DQN, A2C

from environment.custom_env import PharmaStockEnv
from environment.rendering import EnhancedPharmacyRenderer
from training.reinforce_training import REINFORCE


def run_model(algorithm, model_path, num_episodes=5, render=True):
    """Run a trained model"""
    
    print(f"\n{'='*70}")
    print(f"RUNNING {algorithm.upper()} BEST MODEL")
    print(f"{'='*70}")
    print(f"Model: {model_path}")
    print(f"Episodes: {num_episodes}")
    
    env = PharmaStockEnv()
    renderer = None
    
    if render:
        try:
            renderer = EnhancedPharmacyRenderer()
            print("\n Visualization enabled!")
        except Exception as e:
            print(f"\n Warning: Could not initialize renderer: {e}")
            render = False
    
    # Load model based on algorithm
    if algorithm == 'reinforce':
        agent = REINFORCE(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n
        )
        agent.load(model_path)
        print(" Model loaded successfully!")
        
        def get_action(state):
            return agent.select_action(state, deterministic=True)
    else:
        # Stable Baselines 3 models
        if algorithm == 'ppo':
            model = PPO.load(model_path)
        elif algorithm == 'dqn':
            model = DQN.load(model_path)
        elif algorithm == 'a2c':
            model = A2C.load(model_path)
        print(" Model loaded successfully!")
        
        def get_action(state):
            action, _ = model.predict(state, deterministic=True)
            return int(action)
    
    # Run episodes
    for episode in range(num_episodes):
        print(f"\n{'-'*70}")
        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"{'-'*70}")
        
        state, info = env.reset()
        total_reward = 0
        done = False
        step = 0
        
        while not done:
            action = get_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            step += 1
            done = terminated or truncated
            
            if render and renderer:
                render_state = {
                    'stock': info.get('stock', 0),
                    'demand': info.get('demand', 0),
                    'day': info.get('day', step),
                    'action': action,
                    'reward': reward,
                    'stockouts': info.get('stockouts', 0),
                    'service_level': info.get('service_level', 1.0),
                    'pending_order': info.get('pending_order', 0)
                }
                
                if not renderer.render(render_state):
                    print("\nWindow closed by user.")
                    if renderer:
                        renderer.close()
                    return
            
            if step % 5 == 0:
                print(f"  Day {step}: Stock={info.get('stock', 0)}, "
                      f"Demand={info.get('demand', 0)}, Reward={reward:.1f}")
            
            state = next_state
        
        print(f"\n{'='*70}")
        print(f"Episode {episode + 1} Results")
        print(f"{'='*70}")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Service Level: {info.get('service_level', 0)*100:.1f}%")
        print(f"  Stockouts: {info.get('stockouts', 0)}")
        print(f"  Days Completed: {step}")
        
        if episode < num_episodes - 1:
            print("\nStarting next episode in 2 seconds...")
            time.sleep(2)
    
    if renderer:
        renderer.close()
    
    print("\n Demonstration complete!")


def main():
    parser = argparse.ArgumentParser(description='PharmaStock RL - Run trained models')
    parser.add_argument('--algorithm', type=str, default='ppo',
                       choices=['ppo', 'dqn', 'a2c', 'reinforce'],
                       help='Algorithm to run')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to model file')
    parser.add_argument('--episodes', type=int, default=5,
                       help='Number of episodes to run')
    parser.add_argument('--no-render', action='store_true',
                       help='Disable visualization')
    
    args = parser.parse_args()
    
    # Default model paths
    if args.model is None:
        if args.algorithm == 'reinforce':
            args.model = f'models/reinforce/config_10_BEST/best_model.pth'
        else:
            args.model = f'models/{args.algorithm}/config_10_BEST/best_model.zip'
    
    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        print("Please train the model first or specify a valid path.")
        return
    
    run_model(
        algorithm=args.algorithm,
        model_path=args.model,
        num_episodes=args.episodes,
        render=not args.no_render
    )


if __name__ == "__main__":
    main()
