"""
PharmaStock Custom Environment
Gymnasium environment for pharmacy inventory management using Reinforcement Learning
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class PharmaStockEnv(gym.Env):
    """
    Custom Environment for Pharmacy Inventory Management
    
    Mission: Prevent medicine stock-outs while minimizing costs
    
    Agent: AI Inventory Manager
    Goal: Optimize ordering decisions to meet patient demand
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}
    
    def __init__(self, render_mode=None, episode_length=30):
        super().__init__()
        
        self.episode_length = episode_length
        self.render_mode = render_mode
        
        # Action Space: Discrete ordering decisions
        # 0: Order nothing, 1: Order 5 units, 2: Order 10 units, 3: Order 20 units
        self.action_space = spaces.Discrete(4)
        self.order_amounts = [0, 5, 10, 20]
        
        # Observation Space: [current_stock, prev_demand, day, pending_order]
        # All normalized to [0, 1]
        self.observation_space = spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(4,), 
            dtype=np.float32
        )
        
        # Environment parameters
        self.max_stock = 50
        self.min_demand = 2
        self.max_demand = 15
        self.stockout_penalty = 10.0
        self.holding_cost = 0.5
        self.order_cost = 0.2
        
        # State variables
        self.current_stock = 0
        self.day = 0
        self.prev_demand = 0
        self.pending_order = 0
        self.total_stockouts = 0
        self.total_demand_met = 0
        self.total_demand = 0
        
        # Rendering
        self.window = None
        self.clock = None
        
    def _get_obs(self):
        """Return normalized observation"""
        return np.array([
            self.current_stock / self.max_stock,
            self.prev_demand / self.max_demand,
            self.day / self.episode_length,
            self.pending_order / max(self.order_amounts)
        ], dtype=np.float32)
    
    def _get_info(self):
        """Return additional info for logging"""
        return {
            'day': self.day,
            'stock': self.current_stock,
            'demand': self.prev_demand,
            'stockouts': self.total_stockouts,
            'service_level': self.total_demand_met / max(self.total_demand, 1),
            'pending_order': self.pending_order
        }
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        # Initialize state
        self.current_stock = self.np_random.integers(10, 25)
        self.day = 0
        self.prev_demand = self.np_random.integers(self.min_demand, self.max_demand)
        self.pending_order = 0
        self.total_stockouts = 0
        self.total_demand_met = 0
        self.total_demand = 0
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        """Execute one step in the environment"""
        
        # 1. Receive pending order (from previous day)
        self.current_stock += self.pending_order
        self.current_stock = min(self.current_stock, self.max_stock)
        
        # 2. Generate today's demand (with some trend and randomness)
        base_demand = self.np_random.integers(self.min_demand, self.max_demand)
        # Add weekly pattern (higher on weekdays)
        day_of_week = self.day % 7
        if day_of_week < 5:  # Weekday
            demand_multiplier = 1.2
        else:  # Weekend
            demand_multiplier = 0.8
        
        current_demand = int(base_demand * demand_multiplier)
        current_demand = max(self.min_demand, min(current_demand, self.max_demand))
        
        # 3. Try to meet demand
        demand_met = min(self.current_stock, current_demand)
        stockout = max(0, current_demand - self.current_stock)
        
        self.current_stock -= demand_met
        self.total_demand += current_demand
        self.total_demand_met += demand_met
        
        if stockout > 0:
            self.total_stockouts += 1
        
        # 4. Calculate reward
        reward = 0
        
        # Penalty for stock-outs (most important)
        if stockout > 0:
            reward -= self.stockout_penalty * stockout
        
        # Small reward for meeting all demand
        if stockout == 0:
            reward += 2.0
        
        # Penalty for holding too much stock (waste/cost)
        if self.current_stock > 15:
            reward -= self.holding_cost * (self.current_stock - 15)
        
        # Small penalty for ordering (cost)
        order_amount = self.order_amounts[action]
        if order_amount > 0:
            reward -= self.order_cost * order_amount
        
        # 5. Place new order (will arrive next day)
        self.pending_order = order_amount
        
        # 6. Update state
        self.prev_demand = current_demand
        self.day += 1
        
        # 7. Check if episode is done
        terminated = self.day >= self.episode_length
        truncated = False
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment using pygame"""
        if self.render_mode == "human":
            return self._render_frame()
        elif self.render_mode == "rgb_array":
            return self._render_frame(return_rgb=True)
    
    def _render_frame(self, return_rgb=False):
        """Render a single frame using PharmacyRenderer"""
        try:
            from environment.rendering import EnhancedPharmacyRenderer
            
            if self.window is None:
                self.window = EnhancedPharmacyRenderer()
            
            # Prepare state for renderer
            env_state = {
                'stock': self.current_stock,
                'demand': self.prev_demand,
                'day': self.day,
                'action': 0,
                'reward': 0,
                'stockouts': self.total_stockouts,
                'service_level': self.total_demand_met / max(self.total_demand, 1),
                'pending_order': self.pending_order
            }
            
            if return_rgb:
                self.window.render(env_state)
                import pygame
                return np.transpose(
                    np.array(pygame.surfarray.pixels3d(self.window.screen)), axes=(1, 0, 2)
                )
            else:
                return self.window.render(env_state)
            
        except ImportError:
            if self.render_mode == "human":
                print(f"Day {self.day}: Stock={self.current_stock}, "
                      f"Demand={self.prev_demand}, Stockouts={self.total_stockouts}")
    
    def close(self):
        """Clean up resources"""
        if self.window is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()


if __name__ == "__main__":
    print("Testing PharmaStockEnv...")
    
    env = PharmaStockEnv(render_mode="human")
    obs, info = env.reset()
    print(f"\nInitial Observation: {obs}")
    print(f"Initial Info: {info}")
    
    print("\nRunning 10 steps with random actions...")
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {step+1}: Action={env.order_amounts[action]}, Reward={reward:.2f}, Stock={info['stock']}")
        
        if terminated or truncated:
            break
    
    env.close()
    print("\nEnvironment test complete!")
