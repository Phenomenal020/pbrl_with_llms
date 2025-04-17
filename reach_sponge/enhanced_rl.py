# Detailed docstrings are generated using ChatGPT and cross-checked for accuracy

# Some code from : https://github.com/empriselab/RCareWorld/blob/phy-robo-care/pyrcareworld/pyrcareworld/demo/examples/example_rl.py  

import numpy as np
import os
import glob
import random

from generate_poses import GeneratePoses

from perception import Perception

import torch

from on_policy_reward import RewardModel

import jsonlines

try:
    import gymnasium as gym
except ImportError:
    print("This feature requires gymnasium, please install with `pip install gymnasium`")
    raise
from gymnasium import spaces

from pyrcareworld.envs.bathing_env import BathingEnv

from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO

from logger import train_logger

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
        pass  #

os.makedirs("logs/enhanced_rl", exist_ok=True)


class ReachSponge(gym.Env):	
    # Initialisation function
    def __init__(self, use_graphics=False, dev=None, port: int = None):
        super(ReachSponge, self).__init__()
        
        # Initialise the bathing environment - If a port is provided, pass it to BathingEnv.
        if port is not None:
            self.env = BathingEnv(graphics=use_graphics, port=port) if not dev else BathingEnv(graphics=use_graphics, executable_file="@editor", port=port)
        else:
            self.env = BathingEnv(graphics=use_graphics) if not dev else BathingEnv(graphics=use_graphics, executable_file="@editor")
        
        set_global_seed(42)
        
        #TODO: Remove this part later since I do not need it in the actual work (Only foor testing purposes).
        # Grab a perception object to identify the location of objects in the environment
        self.p = Perception(self.env)

        # Get the robot instance in the current environment
        self.robot = self.env.get_robot()
        #  Get the gripper too
        self.gripper = self.env.get_gripper()
        #  Get the sponge
        self.sponge = self.env.get_sponge()
        self.env.step()
        
        # Initialise an n_steps variable to keep track of number of steps taken per episode. 
        self.n_steps = 0
        # Initialise an n_episodes variable to keep track of the total number of episodes
        self.n_episodes = 0
        # Initialise an n_eps_steps to keep track of the total number of steps taken in the current episode
        self.n_eps_steps = 0
        # Track the number of successes
        self.n_success = 0
        
        #  define the action space - Discrete as recommended by the rcareworld authors
        self.action_space = spaces.Discrete(3, start=0)        
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
        
        self.poses = GeneratePoses().train_poses
         
        #  Randomise Robot's initial position and raise the gripper on initialisation
        # For position
        pose = self.np_random.choice(self.poses)
        x, y, z = pose[0][0], pose[0][1], pose[0][2]
        self.robot.SetPosition([x, y, z])
        # For rotation
        xrot, yrot, zrot = pose[1][0], pose[1][1], pose[1][2]
        # Set the robot's rotation
        self.robot.SetRotation([xrot, yrot, zrot])  
        self.env.step(50)
        
        print(f"Position --- {[x, y, z]}")
        print(f"rotation --- {[xrot, yrot, zrot]}")
        
        self.returns = 0

        # temporary buffers to hold data before adding to the reward model
        self.obses = []
        self.actions = []
        self.true_rewards = []
        self.pred_rewards = []
        self.terminateds = []
        self.truncateds = []
        self.collisions = []
        self.successes = []
        
        # for reward shaping
        self.prev_distance = np.linalg.norm(np.array(self.sponge.data.get("position")) - np.array(self.robot.data.get("position")))

        # Initialise reward model to get predicted rewards
        self.reward_model = RewardModel(ds=6, da=3)
        loaded_files = self.reward_model.load(model_dir="reward_model/enhanced_rl/ppo", save_step=5000)
        if not loaded_files:
            raise Exception("No pre-trained model found. A new model will be trained.")
         
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
        pose = self.np_random.choice(self.poses)
        x, y, z = pose[0][0], pose[0][1], pose[0][2]
        self.robot.SetPosition([x, y, z])
        # For rotation
        xrot, yrot, zrot = pose[1][0], pose[1][1], pose[1][2]
        # Set the robot's rotation
        self.robot.SetRotation([xrot, yrot, zrot])  
        self.env.step(50)
        
        print(f"Position --- {[x, y, z]}")
        print(f"rotation --- {[xrot, yrot, zrot]}")
        
        # for reward shaping
        self.prev_distance = np.linalg.norm(np.array(self.sponge.data.get("position")) - np.array(self.robot.data.get("position")))
        
        self.returns = 0
         
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

        # Compute predicted and true rewards based on the new observation
        predicted_reward = self.reward_model.r_hat(observation)
        true_reward, terminated, truncated, n_collision, is_success = self._compute_reward(predicted_reward)
        
        # store data temporarilly (for reward model training)
        self.obses.append(observation)
        self.actions.append(action)
        self.true_rewards.append(true_reward)
        self.pred_rewards.append(predicted_reward)
        self.terminateds.append(terminated)
        self.truncateds.append(truncated)
        self.collisions.append(n_collision)
        self.successes.append(is_success)

        # Check if episode is done (terminated or truncated), then add data in segments of 10 to the reward model.  if episode length < 10, discard. If last segment < 10, combine with penultimate segment, then add of length 10 to the terminated step.
        self.process_episode(terminated, truncated)
        
        info = {}

        # Return the step tuple: observation, predicted_reward, done, and additional info for policy update
        return observation, predicted_reward, terminated, truncated, info
    
    
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


    def process_episode(self, terminated, truncated):
        """
        Process the accumulated episode data. 
        If the episode is terminated or truncated, then add data in segments of length 10 to the reward model.
        If the episode length is less than 10, discard the data.
        If the last segment is incomplete (length < 10), combine it with the penultimate segment, then "slice backwards" and take the last 10 elements to form a segment. Reverse again to get the last segment in the original order.
        """

        # If the episode is terminated or truncated 
        if terminated or truncated: 
            # Get the total number of observations in the episode
            total = len(self.obses)
            # Discard if the total episode length is less than 10. This is definitely a collision without any meaningful progress.
            if total < 10:  # Discard
                # Clear the temporary buffer and return.
                self.obses = []
                self.actions = []
                self.true_rewards = []
                self.pred_rewards = []
                self.terminateds = []
                self.truncateds = []
                self.collisions = []
                self.successes = []
                return

            # Else, 1. get the remainder of the total number of observations in the episode
            remainder = total % 10

            # 2. Helper function to add a segment (size 10) for all data fields.
            def add_segment(start, end):
                self.reward_model.add_data(
                    obs=self.obses[start:end],
                    act=self.actions[start:end],
                    true_rew=self.true_rewards[start:end],
                    pred_rew=self.pred_rewards[start:end],
                    done=self.terminateds[start:end],
                    truncated=self.truncateds[start:end],
                    collision=self.collisions[start:end],
                    success=self.successes[start:end]
                )

            #3. If the data divides evenly into batches of 10, process them normally.
            if remainder == 0:
                for i in range(0, total, 10):
                    add_segment(i, i+10)
            else: # Otherwise,
                #4. If there is an incomplete last segment.
                # Process all complete segments up to the penultimate one.
                # The penultimate full segment starts at index: cutoff = total - remainder - 10.
                cutoff = total - remainder - 10 # (eg, if total = 87, then remainder = 7 and cutoff = 70)
                # Add data normally up to cutoff
                for i in range(0, cutoff, 10):
                    add_segment(i, i+10)

                # Instead of adding the penultimate and the last incomplete segments separately, combine them and adjust by slicing backwards twice.
                def combined_segment(data):
                    # Concatenate penultimate full segment and the incomplete segment.
                    penultimate = data[cutoff:cutoff+10]
                    incomplete = data[cutoff+10: total]
                    combined = penultimate + incomplete  
                    # Reverse the combined list and take the first 10 elements.
                    reversed_list = combined[::-1][:10]
                    # Reverse again to maintain the original order.
                    final_list = reversed_list[::-1]
                    return final_list
                
                # Get the extra segment for each field.
                extra_obs   = combined_segment(self.obses)
                extra_act   = combined_segment(self.actions)
                extra_trew  = combined_segment(self.true_rewards)
                extra_prew  = combined_segment(self.pred_rewards)
                extra_done  = combined_segment(self.terminateds)
                extra_trunc = combined_segment(self.truncateds)
                extra_coll  = combined_segment(self.collisions)
                extra_succ  = combined_segment(self.successes)
                
                # Add the extra segment to the reward model
                self.reward_model.add_data(
                    obs=extra_obs,
                    act=extra_act,
                    true_rew=extra_trew,
                    pred_rew=extra_prew,
                    done=extra_done,
                    truncated=extra_trunc,
                    collision=extra_coll,
                    success=extra_succ
                )

            # Clear the temporary buffers.
            self.obses = []
            self.actions = []
            self.true_rewards = []
            self.pred_rewards = []
            self.terminateds = []
            self.truncateds = []
            self.collisions = []
            self.successes = []

        else:
            return

    
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
        
    
    
    def _compute_reward(self, pred_reward):
        # Check for collisions
        self.env.GetCurrentCollisionPairs()
        self.env.step()
        collision = self.env.data["collision_pairs"]
        
        # Get reward, whether the episode is over, and is_success if the episode was successful
        reward, is_done, truncated, is_success = self._get_reward(collision, pred_reward)
        return reward, is_done, truncated, len(collision), is_success
    
    
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    
    
    def _get_reward(self, collision, pred_reward):
        sponge_pos = self.sponge.data.get("position")
        robot_pos = self.robot.data.get("position")
        robot_rot = self.robot.data.get("rotation")
        self.env.step()
        
        # Reward shaping.
        # Positive reward if the robot moves closer to the goal
        # Penalise timesteps to encourage efficiency
        prev_dist = self.prev_distance  
        curr_dist = np.linalg.norm(np.array(sponge_pos) - np.array(robot_pos))
        dist_reward = (prev_dist - curr_dist) * 10
        time_reward = -0.1 # very small penalty to discourage non-progress
        reward = dist_reward + time_reward
        self.prev_distance = curr_dist
        
        x_low = -2.75
        x_high = 2.75
        z_low = -2.75
        z_high = 2.75

        truncated = False
        is_done = False
        is_success = False 
        
        # Increase the number of steps taken
        self.n_steps = self.n_steps + 1
        # Increase the number of steps taken in the current episode
        self.n_eps_steps = self.n_eps_steps + 1
        
        # If the robot tries to go off-bounds, Penalise the robot and truncate the episode
        # My current action space doesnot have a Move_back action, so out of bounds problems may not be easy to recover from.
        if (robot_pos[0] > x_high or robot_pos[0] < x_low or robot_pos[2] > z_high or robot_pos[2] < z_low):
            reward = reward - 10
            truncated = True
            
        # Penalise the robot and end the episode if there is a collision
        # The robot might get stuck or fall. Punish this severely.
        if len(collision) > 0:
            reward = reward - 10
            is_done = True
            
        # Check if the robot is in the goal area
        full_goal, partial_goal = self.is_in_goal_area(robot_pos, robot_rot)
        if full_goal:
            print("------ in goal area with correct gripper alignment ------")
            reward = reward + 30
            self.n_success += 1
            is_success = True
            is_done = True
        if partial_goal:
            print("------ in goal area with incorrect gripper alignment ------")
            reward = reward + 20
            is_success = True
            is_done = True
            self.n_success += 1
            
        # Truncate after 100 steps per episode
        if self.n_eps_steps >= 100:
            truncated = True
            reward += -2 * np.linalg.norm(np.array(sponge_pos) - np.array(robot_pos))
            
        self.returns += reward
        
        print({
            "n_steps": self.n_steps,
            "num_episodes": self.n_episodes,
            "num_episodes_steps": self.n_eps_steps,
            "true reward": reward,
            "pred_reward" : pred_reward,
            "done": is_done,
            "truncated": truncated,
            "n_collisions": len(collision),
            "n_successes": self.n_success,
            "returns": self.returns
        })  
       
        with jsonlines.open("logs/enhanced_rl/train.jsonl", mode="a") as writer:
            writer.write({
                # "obs": obs,
                # "action": action,
                "true_reward": reward,
                "pred_reward": pred_reward,
                "is_done": is_done,
                "truncated": truncated,
                "collision": collision,
                "success": is_success
            })
        print(f"Results saved to file.")

        
        return (reward, is_done, truncated, is_success)
    
    
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



    # Check if the robot is in an easy grasp position
    def is_in_goal_area(self, robot_pos, robot_rot):
        # Define the goal area bounds
        xmax, xmin = -0.050, -0.258
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
              
              
                   
if __name__ == "__main__":
        
    def make_env(port):
        env = ReachSponge(use_graphics=False, port=port)  
        env = Monitor(env, filename=f"enhanced_reach_sponge_monitor_{port}.csv")
        return env

    # Manually specify the ports to use for each environment
    # manual_ports = [5005, 5261, 5517, 5750, 5801, 5823, 5837, 5851, 5875, 5899]
    manual_ports = [5005]

    # Create a list of lambda functions for each environment, each with its assigned port
    env_fns = [lambda port=port: make_env(port) for port in manual_ports]
    env = SubprocVecEnv(env_fns)
        
    # Directories for models and logs
    models_dir = "models/enhanced_rl/ppo"
    logdir = "logs/enhanced_rl/ppo"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logdir, exist_ok=True)

    # Check for existing saved models (checkpoints)
    latest_checkpoint = None
    latest_timestep = 0
    checkpoint_files = glob.glob(os.path.join(models_dir, "enhanced_reach_sponge_ppo*.zip"))
        
     # Loop through each checkpoint file and update the latest timestep variable to the last checkpoint file's save timestep
    for f in checkpoint_files:
        basename = os.path.basename(f)
        try:
            # Remove filename parts and retrieve SAVE_INTERVAL timestep
            timestep = int(basename.replace("enhanced_reach_sponge_ppo", "").replace(".zip", ""))
            if timestep > latest_timestep:
                latest_timestep = timestep
                latest_checkpoint = f
        except ValueError:
            continue
        
    # If a checkpoint exists, load it. Else, create a new model.
    if latest_checkpoint is not None:
        print(f"Resuming model training from checkpoint: {latest_checkpoint}")
        model = PPO.load(latest_checkpoint, env=env, tensorboard_log=logdir, n_steps=1024)
    else:
        print("No checkpoint found. Initialising a new model.")
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)
        
    # Training parameters
    TOTAL_TIMESTEPS = 100000  
    SAVE_INTERVAL = 1000
    REMAINING_TIMESTEPS = TOTAL_TIMESTEPS - latest_timestep
        
    # Incremental training loop starting from the latest saved timestep
    for i in range(latest_timestep, latest_timestep + REMAINING_TIMESTEPS, SAVE_INTERVAL):
        model.learn(total_timesteps=SAVE_INTERVAL,
                    reset_num_timesteps=False,
                    tb_log_name="ppo")
            
        # Save the model checkpoint with the cumulative timestep count
        cumulative_timestep = i + SAVE_INTERVAL
        save_path = os.path.join(models_dir, f"enhanced_reach_sponge_ppo{cumulative_timestep}.zip")
        model.save(save_path)
        print(f"Model saved at timestep {cumulative_timestep}")
        
    # Close the environment after training
    env.close()