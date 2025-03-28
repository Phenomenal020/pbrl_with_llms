import os
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv
import json
import numpy as np  # Make sure to import numpy for type checking
import random

# Custom encoder for NumPy types.
def numpy_encoder(obj):
    if isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

# Define a simple prompt formatter.
def format_traj_for_gpt(traj1, traj2):
    obs1, acts1, r1 = traj1
    obs2, acts2, r2 = traj2

    # Convert each component to a Python list if possible.
    traj1_data = {
        "observations": obs1.tolist() if hasattr(obs1, "tolist") else obs1,
        "actions": acts1.tolist() if hasattr(acts1, "tolist") else acts1,
        "rewards": r1.tolist() if hasattr(r1, "tolist") else r1,
    }
    traj2_data = {
        "observations": obs2.tolist() if hasattr(obs2, "tolist") else obs2,
        "actions": acts2.tolist() if hasattr(acts2, "tolist") else acts2,
        "rewards": r2.tolist() if hasattr(r2, "tolist") else r2,
    }

    # Use the custom numpy_encoder in json.dumps.
    traj1_str = json.dumps(traj1_data, indent=2, default=numpy_encoder)
    traj2_str = json.dumps(traj2_data, indent=2, default=numpy_encoder)

    prompt = (
        "I have two trajectories from a FrozenLake environment. "
        "Each trajectory consists of a list of observations, actions taken, and received rewards.\n\n"
        "Trajectory 1:\n"
        f"{traj1_str}\n\n"
        "Trajectory 2:\n"
        f"{traj2_str}\n\n"
        "Based on the data provided, which trajectory seems to be more successful? "
        "Please respond with '1' if Trajectory 1 is better, or '2' if Trajectory 2 is better."
    )
    return prompt

class CustomFrozenLakeEnv(FrozenLakeEnv):
    def __init__(self, **kwargs):
        # Pass render_mode="human" so the environment creates a window.
        super().__init__(**kwargs, render_mode="human")
        self.r1 = []
        self.r2 = []
        self.acts1 = []
        self.acts2 = []
        self.obs1 = []
        self.obs2 = []
        self.nsteps = 0

    # Override step to collect data.
    def step(self, action):
        # Call the original step method.
        obs, reward, terminated, truncated, info = super().step(action)
        reward = +obs if (not terminated and not truncated) else -5
        self.r1.append(reward)
        self.acts1.append(action)
        self.obs1.append(obs)
        self.r2.append(reward)
        self.acts2.append(action)
        self.obs2.append(obs)
        self.nsteps += 1
        return obs, reward, terminated, truncated, info

    # Sample a window of traj length 10 from each buffer.
    def get_data(self):
        # Ensure there are enough data points.
        if len(self.obs1) < 10 or len(self.obs2) < 10:
            raise ValueError("Not enough data to sample a trajectory of length 10.")
        
        max_index1 = len(self.obs1) - 10
        max_index2 = len(self.obs2) - 10
        
        start_index1 = random.randint(0, max_index1)
        start_index2 = random.randint(0, max_index2)
        
        # Sample a window of length 10 for each trajectory.
        traj1 = (
            self.obs1[start_index1:start_index1+10],
            self.acts1[start_index1:start_index1+10],
            self.r1[start_index1:start_index1+10]
        )
        traj2 = (
            self.obs2[start_index2:start_index2+10],
            self.acts2[start_index2:start_index2+10],
            self.r2[start_index2:start_index2+10]
        )
        
        return traj1, traj2

def main():
    # Create the custom FrozenLake environment.
    env = CustomFrozenLakeEnv(is_slippery=False)
    obs, _ = env.reset()
    
    done = False
    truncated = False
    while env.nsteps < 100:
        if not done and not truncated:
            # For this test, choose random actions.
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
        else:
            env.reset()
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            
            # Render the environment to display its current state.
        env.render()
        print(f"Step {env.nsteps}: obs={obs}, reward={reward}")
    
    # Once 100 steps are collected, sample two trajectory windows.
    traj1, traj2 = env.get_data()
    
    # Format the prompt with the two trajectories.
    prompt = format_traj_for_gpt(traj1, traj2)
    print("GPT Query Prompt:")
    print(prompt)
    
    # Optionally write the prompt to a file.
    with open("prompt.txt", "w") as f:
        f.write(prompt)

if __name__ == '__main__':
    main()