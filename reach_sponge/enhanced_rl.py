# Detailed docstrings are generated using ChatGPT and cross-checked for accuracy

# Some code from : https://github.com/empriselab/RCareWorld/blob/phy-robo-care/pyrcareworld/pyrcareworld/demo/examples/example_rl.py  

# TODO: Import these
from CustomDQN import CustomDQN
from CustomRB import DQNPBRLReplayBuffer
from reward import RewardModel

import numpy as np
import torch
import os
import glob
import random

from generate_poses import GeneratePoses

from perception import Perception

try:
    import gymnasium as gym
except ImportError:
    print("This feature requires gymnasium, please install with `pip install gymnasium`")
    raise
from gymnasium import spaces

from pyrcareworld.envs.bathing_env import BathingEnv
import pyrcareworld.attributes as attr
from pyrcareworld.envs.base_env import RCareWorld

# Import DQN and other utilities from stable_baselines3
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.dqn.policies import MlpPolicy
from stable_baselines3.common.monitor import Monitor


# Ensure required directories exist.
os.makedirs("./enhanced_rl/models", exist_ok=True)
os.makedirs("./enhanced_rl/tensorboard_logs", exist_ok=True)
os.makedirs("./enhanced_rl/trajectories", exist_ok=True)
os.makedirs("./enhanced_rl/logs", exist_ok=True)


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            print("cuda available")
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        print("No cuda available")
        pass  


class ReachSponge(gym.Env):
    
    # Initialisation function
    def __init__(self, use_graphics=False, dev=None, total_timesteps=10000, save_interval=500, model_file=None):
        super(ReachSponge, self).__init__()

        # Initialise the bathing environment in headless mode 
        self.env = BathingEnv(graphics=use_graphics) if not dev else BathingEnv(graphics=use_graphics, executable_file="@editor")
        set_global_seed(42)

        #TODO: Remove this part later since I do not need it in the actual work.
        # Grab a perception object to identify the location of objects in the environment
        p = Perception(self.env)

        # Get the robot instance in the current environment
        self.robot = self.env.get_robot()
        #  Get the gripper too
        self.gripper = self.env.get_gripper()
        #  Get the sponge
        self.sponge = self.env.get_sponge()
        self.env.step()
        
        # Initialise an n_steps variable to keep track of the total number of steps taken
        self.n_steps = 0
        # Initialise an n_episodes variable to keep track of the total number of episodes
        self.n_episodes = 0
        # Initialise an n_eps_steps to keep track of the total number of steps taken in the current episode
        self.n_eps_steps = 0
        
        # Temporary buffers to store trajectory data before adding to the reward model
        self.obs_buffer = []
        self.action_buffer = []
        self.true_reward_buffer = []
        self.predicted_reward_buffer = []
        self.dones_buffer = []
        self.truncated_buffer = []
        
        #  define the action space - Discrete as recommended by the rcareworld authors
        self.action_space = spaces.Discrete(2)        
        # define the observation space. The robot's position should be in the goal area. The orientation is important to properly align the gripper
        # Flatten the observation space
        low = np.concatenate([
            np.array([-5.0, -2.0, -5.0]),          # robot_position
            np.array([-360, -360, -360])           # robot_rotation
        ])
        high = np.concatenate([
            np.array([5.0, 2.0, 5.0]),             # robot_position
            np.array([360, 360, 360])              # robot_rotation
        ])
        # Define the flattened observation space.
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        # Initialise a reward model
        self.reward_model = RewardModel(
            ds=self.observation_space.shape[0],
            da=self.action_space.n,
            teacher_beta=1,
            teacher_gamma=0.99,
            teacher_eps_mistake=0,
            teacher_eps_skip=0,
            teacher_eps_equal=0,
            save_equal=True,
            traj_action=False,
            traj_save_path="./enhanced_rl/trajectories",
        )
        
        # Create a replay buffer for a discrete observation and action space.
        self.replay_buffer = DQNPBRLReplayBuffer(
            buffer_size=100000,
            observation_space=self.observation_space,
            action_space=self.action_space
        )
        
        # Generate training data and initialise robot pose
        self.poses = GeneratePoses().train_poses
        
        #  Randomise Robot's initial position and raise the gripper on initialisation
        # For position
        pose = random.choice(self.poses)
        x, y, z = pose[0][0], pose[0][1], pose[0][2]
        self.robot.SetPosition([x, y, z])
        # For rotation
        xrot, yrot, zrot = pose[1][0], pose[1][1], pose[1][2]
        # Set the robot's rotation
        self.robot.SetRotation([xrot, yrot, zrot])  
        self.env.step(50)
        
        print(f"Position --- {[x, y, z]}")
        print(f"rotation --- {[xrot, yrot, zrot]}")
         
        # Raise the gripper to a safe height for easy obstacle avoidance
        self.gripper.GripperOpen()
        self.robot.WaitDo()
        self.env.step()
            
        gripper_pos = [self.robot.data["position"][0], self.sponge.data.get("position")[1] + 0.3, self.robot.data["position"][2]]
        self.robot.IKTargetDoMove(
            position=gripper_pos,
            duration=1,
            speed_based=False,
        )
        self.robot.WaitDo()
        self.env.step()
        
      
    #   +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      
      
      
    # Whenever the environment is reset (after an episode), initialise the robot, gripper, and return an observation 
    def reset(self, seed=None, options=None):
        # Handle the seed if provided - for reproducibility
        if seed is not None:
            self.np_random, seed = gym.utils.seeding.np_random(seed)
            
        # Increase n_episodes counter
        self.n_episodes += 1
        # Reset n_eps_steps counter
        self.n_eps_steps = 0
        
        #  Randomise Robot's initial position and raise the gripper on initialisation
        # For position
        pose = random.choice(self.poses)
        x, y, z = pose[0][0], pose[0][1], pose[0][2]
        self.robot.SetPosition([x, y, z])
        # For rotation
        xrot, yrot, zrot = pose[1][0], pose[1][1], pose[1][2]
        # Set the robot's rotation
        self.robot.SetRotation([xrot, yrot, zrot])  
        self.env.step(50)
        
        print(f"Position --- {[x, y, z]}")
        print(f"rotation --- {[xrot, yrot, zrot]}")
         
        # Raise the gripper to a safe height for easy obstacle avoidance
        self.gripper.GripperOpen()
        self.robot.WaitDo()
        self.env.step(50)
            
        gripper_pos = [self.robot.data["position"][0], self.sponge.data.get("position")[1] + 0.3, self.robot.data["position"][2]]
        self.robot.IKTargetDoMove(
            position=gripper_pos,
            duration=1,
            speed_based=False,
        )
        self.robot.WaitDo()
        self.env.step(50)
        
        observation = self._get_observation()
        print("++++++++++++++")
        print("resetting")
        return observation, {} 
    
    
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    
    
    # Each step in an episode
    def step(self, action):
        # Apply the selected action
        self._perform_action(action)

        # Get the updated observation
        observation = self._get_observation()
        
        # Compute the predicted reward using the reward model
        predicted_reward = self.reward_model.r_hat(observation)

        # Compute the true reward based on the new state
        true_reward, terminated, truncated = self._compute_reward(predicted_reward)
        
        # Add the current step to the temporary buffers
        self.obs_buffer.append(observation)
        self.action_buffer.append(action)
        self.true_reward_buffer.append(true_reward)
        self.predicted_reward_buffer.append(predicted_reward)
        self.dones_buffer.append(terminated)
        self.truncated_buffer.append(truncated)
        
        # If the episode has ended (terminated or truncated),
        # move the entire temporary buffer to the reward model and clear it.
        if terminated or truncated:
            self.reward_model.add_data(
                self.obs_buffer, 
                self.action_buffer, 
                self.true_reward_buffer, 
                self.predicted_reward_buffer, 
                self.dones_buffer,
                self.truncated_buffer
            )
            self.obs_buffer = []
            self.action_buffer = []
            self.true_reward_buffer = []
            self.predicted_reward_buffer = []
            self.dones_buffer = []
            self.truncated_buffer = []

        info = {}
        # Return the step tuple for the RL algorithm
        return observation, predicted_reward, terminated, truncated, info

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    
    
    def render(self):
        pass
    
    
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    
    
    def close(self):
        self.env.close()
        
        
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        
    
    
    # Helper methods 
    # ------------------------------------------------------------------------------
    
    def _perform_action(self, action):
        # Move forward
        if action == 0:  
            self.robot.MoveForward(0.2, 2)
            print("Robot moving forward")
            self.env.step(50)
        # Turn left
        elif action == 1:  
            self.robot.TurnLeft(90, 1.5)
            print("Robot turning left")
            self.env.step(225)
        # Turn right
        elif action == 2:  
            self.robot.TurnRight(90, 1.5)
            print("Robot turning right")
            self.env.step(225)
        else:
            pass
        
    
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        
        
        
    def _get_observation(self):
        robot_position = np.array(self.robot.data.get("position"), dtype=np.float32)
        robot_rotation = np.array(self.robot.data.get("rotation"), dtype=np.float32)
        
        # Concatenate all components into a single flat array.
        flattened_observation = np.concatenate([
            robot_position, 
            robot_rotation
        ])
        return flattened_observation
        
    
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        
    
    
    def _compute_reward(self, predicted_reward):
        # Check for collisions
        self.env.GetCurrentCollisionPairs()
        self.env.step()
        collision = self.env.data["collision_pairs"]
        
        # Get reward and whether the episode is over
        reward, is_done, truncated = self._get_reward(collision, predicted_reward)
        return reward, is_done, truncated
    
    
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    
    
    def _get_reward(self, collision, predicted_reward):
        sponge_pos = self.sponge.data.get("position")
        robot_pos = self.robot.data.get("position")
        robot_rot = self.robot.data.get("rotation")
        robot_x, robot_z = robot_pos[0], robot_pos[2]
        sponge_x, sponge_z = sponge_pos[0], sponge_pos[2]
        
        # Alternatively;
        dist = np.sqrt((sponge_x - robot_x)**2 + (sponge_z - robot_z)**2)
        reward = -dist
        
        is_done = False
        x_low = -2.5
        x_high = 2.5
        z_low = -3.0
        z_high = 3.0

        self.env.step()
        
        truncated = False
        
        # Increase the number of steps taken
        self.n_steps = self.n_steps + 1
        # Increase the number of steps taken in the current episode
        self.n_eps_steps = self.n_eps_steps + 1
        
        # If the robot tries to go off-bounds, Penalise the robot and truncate the episode
        if (robot_pos[0] > x_high or robot_pos[0] < x_low or robot_pos[2] > z_high or robot_pos[2] < z_low):
            reward = reward - 5
            truncated = True
            
        # Penalise the robot and end the episode if there is a collision
        if len(collision) > 0:
            reward = reward - 10
            is_done = True
            
        # Check if the robot is in the goal area
        full_goal, partial_goal = self.is_in_goal_area(robot_pos, robot_rot)
        if full_goal:
            print("------ in goal area with correct gripper alignment ------")
            reward = reward + 20
            is_done = True
        if partial_goal:
            print("------ in goal area with incorrect gripper alignment ------")
            reward = reward + 10
            is_done = True
            
        # Truncate after 100 steps per episode
        if self.n_steps >= 100:
            truncated = True
            is_done = True
            
        print({
            "n_steps": self.n_steps,
            "num_episodes": self.n_episodes,
            "num_episodes_steps": self.n_eps_steps,
            "true reward": reward,
            "predicted reward": predicted_reward,
            "done": is_done,
            "truncated": truncated
        })   
        
        return (reward, is_done, truncated)
    
    
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



    # Check if the robot is in an easy grasp position
    def is_in_goal_area(self, robot_pos, robot_rot):
        # Define the goal area bounds
        xmax, xmin = -0.050, -0.259
        # ymin, ymax = 0, 0  # ignore 
        zmax, zmin = 1.725, 1.400
        
        # Define valid rotation ranges
        xrot = {355, 356, 357, 358, 359}
        yrot = {260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280}
        zrot = {0, 358, 359, 360}
        
        # Extract robot position and rotation
        x = robot_pos[0]
        y = robot_pos[1]
        z = robot_pos[2]
        rx = int(robot_rot[0])
        ry = int(robot_rot[1])
        rz = int(robot_rot[2])
        
        # Check position constraints
        valid_pos = xmin <= x <= xmax and zmin <= z <= zmax
        
        # Check rotation constraints
        valid_rot = rx in xrot and ry in yrot and rz in zrot
        
        
        return (valid_pos and valid_rot, valid_pos and not valid_rot)
            
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
              

# Assume the following custom classes are properly imported:
# - ReachSponge
# - CustomDQN

def ensure_directories(dirs):
    """Ensure each directory in dirs exists."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        

def make_env(model_file):
    """Create and return a monitored environment."""
    env = ReachSponge(total_timesteps=50000, save_interval=500, model_file=model_file)
    env = Monitor(env, filename="enhanced_reach_sponge_monitor.csv")
    return env


def get_latest_checkpoint(models_dir, prefix="dqn_reach_sponge_enhanced", ext=".zip"):
    """Return the path and timestep of the latest checkpoint file, or (None, 0)."""
    latest_timestep = 0
    latest_checkpoint = None
    pattern = os.path.join(models_dir, f"{prefix}*{ext}")
    for f in glob.glob(pattern):
        basename = os.path.basename(f)
        try:
            timestep_str = basename.replace(prefix, "").replace(ext, "")
            timestep = int(timestep_str)
            if timestep > latest_timestep:
                latest_timestep = timestep
                latest_checkpoint = f
        except ValueError:
            continue
    return latest_checkpoint, latest_timestep


def get_latest_reward_model(reward_model_dir, prefix="reward_model", ext=".pt"):
    """Return the path and timestep of the latest reward model file, or (None, 0)."""
    latest_reward_timestep = 0
    latest_reward_model = None
    pattern = os.path.join(reward_model_dir, f"{prefix}*{ext}")
    for f in glob.glob(pattern):
        basename = os.path.basename(f)
        try:
            timestep_str = basename.replace(prefix, "").replace(ext, "")
            timestep = int(timestep_str)
            if timestep > latest_reward_timestep:
                latest_reward_timestep = timestep
                latest_reward_model = f
        except ValueError:
            continue
    return latest_reward_model, latest_reward_timestep


def main():
    # Define useful directories and file paths.
    tensorboard_log_dir = "./enhanced_rl/tensorboard_logs"
    model_file = "./enhanced_rl/models/dqn_reach_sponge_enhanced.zip"
    models_dir = "./enhanced_rl/models"
    reward_model_dir = "./enhanced_rl/reward_models"
    trajectory_dir = "./enhanced_rl/trajectories"
    
    # Ensure all required directories exist.
    ensure_directories([tensorboard_log_dir, models_dir, reward_model_dir, trajectory_dir])
    
    # Create a vectorized environment.
    env = DummyVecEnv([lambda: make_env(model_file)])
    
    # Check for saved checkpoints and reward models.
    latest_checkpoint, latest_timestep = get_latest_checkpoint(models_dir)
    latest_reward_model, latest_reward_timestep = get_latest_reward_model(reward_model_dir)
    
    # Load a model based on available saved files.
    if latest_checkpoint is not None:
        print(f"Resuming training from checkpoint: {latest_checkpoint}")
        model = CustomDQN.load(latest_checkpoint, env=env, tensorboard_log=tensorboard_log_dir)
    elif latest_reward_model is not None:
        print(f"Resuming training from reward model: {latest_reward_model}")
        model = CustomDQN.load(latest_reward_model, env=env, tensorboard_log=tensorboard_log_dir)
    elif os.path.exists(model_file):
        print("Model file found. Resuming training from the saved model.")
        model = CustomDQN.load(model_file, env=env, tensorboard_log=tensorboard_log_dir)
    else:
        print("No saved model found. Creating a new model.")
        model = CustomDQN(
            learning_starts=500,
            env=env,
            policy="MlpPolicy",
            reward_model=env.reward_model,
            replay_buffer=env.replay_buffer,
            tensorboard_log=tensorboard_log_dir,
            train_freq=(10, "episode"),
        )
    
    # Training parameters.
    TOTAL_TIMESTEPS = 50000  
    SAVE_INTERVAL = 500
    remaining_timesteps = TOTAL_TIMESTEPS - latest_timestep
    
    # Incremental training loop in SAVE_INTERVAL increments.
    for i in range(latest_timestep, latest_timestep + remaining_timesteps, SAVE_INTERVAL):
        model.learn(total_timesteps=SAVE_INTERVAL, reset_num_timesteps=False, tb_log_name="dqn")
        # Save reward model (both checkpoint and full model).
        env.reward_model._save(model_dir=reward_model_dir, step=i + SAVE_INTERVAL)
        env.reward_model.save()  # Saving the full reward model.
        # Save the main model checkpoint.
        save_path = os.path.join(models_dir, f"dqn_reach_sponge_enhanced{i + SAVE_INTERVAL}.zip")
        model.save(save_path)
        print(f"Model saved at timestep {i + SAVE_INTERVAL}")
    
    # Clean up the environment.
    env.close()


if __name__ == "__main__":
    main()