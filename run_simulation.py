import torch
import numpy as np
import time
import os
import sys
from stable_baselines3 import PPO
from environment.custom_env import PharmaStockEnv
from environment.rendering import EnhancedPharmacyRenderer 
from training.reinforce_training import PolicyNetwork, REINFORCE

def run_simulation(model_path):
    """
    Runs a live simulation of the trained agent with HD graphics.
    """
    print(f"Loading model from: {model_path}")
    
    # 1. Initialize Environment and Renderer
    env = PharmaStockEnv()
    renderer = EnhancedPharmacyRenderer()
    
    # 2. Initialize Agent and Load Weights
    ppo_model = None
    reinforce_agent = None

    if model_path.endswith(".zip"):
        # Load Stable Baselines 3 PPO Model
        if os.path.exists(model_path):
            ppo_model = PPO.load(model_path)
            print("✅ PPO Model loaded successfully (Stable Baselines 3)!")
        else:
             print(f"❌ Error: Model file not found at {model_path}")
             return
    else:
        # Load Custom REINFORCE Model
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
        reinforce_agent = REINFORCE(state_dim, action_dim)
        
        if os.path.exists(model_path):
            reinforce_agent.load(model_path)
            print("✅ REINFORCE Model loaded successfully!")
        else:
            print(f"❌ Error: Model file not found at {model_path}")
            return

    # 3. Run the Simulation Loop
    print("\nStarting Simulation... (Press Close Window to Stop)")
    
    state, _ = env.reset()
    done = False
    total_reward = 0
    step = 0
    
    running = True
    while running:
        if ppo_model:
            # PPO prediction (deterministic=True is best for evaluation)
            action, _ = ppo_model.predict(state, deterministic=True)
            action = int(action) # Convert numpy int to python int
        else:
            # REINFORCE prediction
            action = reinforce_agent.select_action(state, deterministic=True)
        
        # Environment takes step
        next_state, reward, terminated, truncated, info = env.step(action)
        
        # Update visualization
        # Construct the state dictionary expected by the renderer
        render_state = {
            'stock': info.get('stock', 0),         # Get from info or state
            'demand': info.get('demand', 0),
            'day': step,
            'action': action,
            'reward': reward,
            'stockouts': info.get('stockouts', 0),
            'service_level': info.get('service_level', 0),
            'pending_order': info.get('pending_order', 0)
        }
        
        # Render frame
        running = renderer.render(render_state)
        
        # Move to next state
        state = next_state
        total_reward += reward
        step += 1
        done = terminated or truncated
        
        # Slow down slightly so we can see what's happening
        # time.sleep(0.1) 
        
        if done:
            print(f"Episode Finished! Total Reward: {total_reward:.2f}")
            # Optional: Reset to keep watching indefinitely
            state, _ = env.reset()
            done = False
            total_reward = 0
            step = 0
            time.sleep(1) # Pause briefly between episodes

    renderer.close()
    print("Simulation closed.")

if __name__ == "__main__":
    MODEL_PATH = "models/ppo/config_10_BEST/best_model.zip"
    
    run_simulation(MODEL_PATH)
