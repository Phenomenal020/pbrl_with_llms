# Detailed docstrings are generated using ChatGPT and cross-checked for accuracy

# Some code from : https://github.com/empriselab/RCareWorld/blob/phy-robo-care/pyrcareworld/pyrcareworld/demo/examples/example_rl.py  

import numpy as np
import os
import glob
import random
import cv2

from datetime import datetime

from generate_poses import GeneratePoses

from perception import Perception

import torch
import pyrcareworld.attributes.camera_attr as attr

# from off_policy_reward import RewardModel

import jsonlines

os.makedirs("ablation/images/dqn", exist_ok=True)

try:
    import gymnasium as gym
except ImportError:
    print("This feature requires gymnasium, please install with `pip install gymnasium`")
    raise
from gymnasium import spaces

from pyrcareworld.envs.bathing_env import BathingEnv

from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnvWrapper
from stable_baselines3 import DQN

from off_policy_reward import RewardModel
from CustomRB import DQNPBRLReplayBuffer

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

# Make sure these directories exist - logs dir, enhanced rb, enhanced rm, enhanced model
os.makedirs("enhanced_rl/logs/dqn", exist_ok=True)
os.makedirs("enhanced_rl/replay_buffer/dqn", exist_ok=True)
os.makedirs("enhanced_rl/reward_model/dqn", exist_ok=True)
os.makedirs("enhanced_rl/model/dqn", exist_ok=True)

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
        
        self.port = port
        # self.p = Perception(self.env)
        
        
        
        # Setup a camera to capture the scene images for ablation 
        self.env.scene_camera = self.env.InstanceObject(name="Camera", id=123456, attr_type=attr.CameraAttr) 
        self.env.scene_camera.SetTransform(position=[0, 4.2, 0.6], rotation=[90, 0, 0.0])
        self.env.AlignCamera(123456)
        self.env.step()
        
        # setup directories
        self.log_dir = "enhanced_rl/logs/dqn"
        self.model_dir = "enhanced_rl/model/dqn"
        self.reward_model_dir = "enhanced_rl/reward_model/dqn"
        self.replay_buffer_dir = "enhanced_rl/replay_buffer/dqn"

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
        # TRack number of turns in current episode to discourage excessive turning
        self.n_turns = 0
        
        self.counter = -1
        
        self.image = None
        
        #  define the action space - Discrete as recommended by the rcareworld authors
        self.action_space = spaces.Discrete(3, start=0)  
              
        # define the observation space.
        low = np.concatenate([
            np.array([-5.0, -5.0, -360, -5.0, -5.0, -5.0, -5.0]) 
        ])  
        high = np.concatenate([
            np.array([5.0, 5.0, 360, 5.0, 5.0, 5.0, 5.0])              # robot_rotation
        ]) # [robotdX, robotdZ, robotrY, robotd2bedx, robotd2bedz, robotd2drawerx, robotd2drawerz]
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        # Obstacles in the environment - from the perception system
        self.bedXL = -1.275
        self.bedXR =  1.275
        self.bedZT = 0.775
        self.bedZB = -0.775
        self.drawerXL = -0.315
        self.drawerXR = 0.638
        self.drawerZT = 2.775
        self.drawerZB = 1.726
        # sponge_location - goal position
        self.sponge_position = np.array(self.sponge.data.get("position"), dtype=np.float32)
        self.env.step()
        
        self.poses = GeneratePoses().test_poses
        #  Randomise Robot's initial position and raise the gripper on initialisation
        # For position
        pose = self.np_random.choice(self.poses)
        # pose  = self.poses[self.counter]
        x, y, z = pose[0][0], pose[0][1], pose[0][2]
        self.robot.SetPosition([x, y, z])
        # For rotation
        xrot, yrot, zrot = pose[1][0], pose[1][1], pose[1][2]
        # Set the robot's rotations
        self.robot.SetRotation([xrot, yrot, zrot])  
        self.env.step(50)
        
        print(f"Position --- {[x, y, z]}")
        print(f"rotation --- {[xrot, yrot, zrot]}")
        
        self.imagedir = "ablation/images/dqn"
        
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
        self.images = []
        
        # Directory for data storage
        self.file_db = f"ablation/rm/trajectories/enhanced_reach_sponge{port}64k.jsonl"
        os.makedirs(os.path.dirname(self.file_db), exist_ok=True)
        
        # for reward shaping
        self.prev_distance = np.linalg.norm(np.array(self.sponge.data.get("position")) - np.array(self.robot.data.get("position")))
        
        # Initialise reward model to get predicted rewards
        self.load_step = 64000
        self.reward_model = RewardModel(ds=7, da=3)
        loaded_files = self.reward_model.load(model_dir="enhanced_rl/reward_model/dqn", load_step=64000)
        if not loaded_files:
            raise Exception("No reward model found. A new model will be trained.")
         
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
            
        #  Randomise Robot's initial position and raise the gripper on initialisation
        # For position
        # pose = self.np_random.choice(self.poses)
        pose = self.poses[self.counter]
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
        self.counter += 1
        return observation, {}
    
    
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    
    # Each step in an episode
    def step(self, action):
        # Apply the selected action
        self._perform_action(action)

        # Get the updated observation
        observation = self._get_observation()

        # Compute predicted and true rewards based on the new observation
        predicted_reward = self.reward_model.r_hat(np.array(observation))
        predicted_reward = predicted_reward.detach().cpu().item()
        true_reward, terminated, truncated, n_collision, is_success = self._compute_reward(predicted_reward)
        
        # capture scene image
        image = self.env.scene_camera.GetRGB(width=512, height=512)
        self.env.step()
        rgb = np.frombuffer(self.env.scene_camera.data["rgb"], dtype=np.uint8)
        image = cv2.imdecode(rgb, cv2.IMREAD_COLOR)
        # cv2.imshow("scene", image)
        # cv2.waitKey(0)              # wait indefinitely until a key is pressed
        # cv2.destroyAllWindows() 
        if image is None:
            print("Error: Decoded image is None.")
            return
        else:
            print("Image successfully captured. Image shape:", image.shape) 
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}{self.port}.png"
        print(f"filename: {filename}")
        cv2.imwrite(os.path.join(self.imagedir, filename), image)
        
        # store data temporarilly (for reward model training)
        self.obses.append(observation)
        self.actions.append(action)
        self.true_rewards.append(true_reward)
        self.pred_rewards.append(predicted_reward)
        self.terminateds.append(terminated)
        self.truncateds.append(truncated)
        self.collisions.append(n_collision)
        self.successes.append(is_success)
        self.images.append(filename)

        # Check if episode is done (terminated or truncated), then add data in segments of 5 to the reward model.  if episode length < 5, discard. If last segment < 5, combine with penultimate segment, then add of length 10 to the terminated step.
        self.process_episode(terminated, truncated)
                    
        if terminated or truncated:
            self.n_episodes += 1
            self.n_eps_steps = 0
            self.n_turns = 0
        
        info = {}

        # Return the step tuple: observation, predicted_reward, done, and additional info for policy update
        return observation, predicted_reward, terminated, truncated, info
    
    
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    def format_json(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.generic,)):
            return obj.item()
        elif isinstance(obj, (list, tuple)):
            return [self.format_json(x) for x in obj]
        elif isinstance(obj, dict):
            return {k: self.format_json(v) for k, v in obj.items()}
        else:
            return obj


    def process_episode(self, terminated, truncated):

        # If the episode is terminated or truncated 
        if terminated or truncated: 
            # Get the total number of observations in the episode
            total = len(self.obses)
            # Discard if the total episode length is less than 5. This is definitely a collision without any meaningful progress.
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
                self.images = []
                return

            # Else, 1. get the remainder of the total number of observations in the episode
            remainder = total % 10

            # 2. Helper function to add a segment (size 10) for all data fields.
            def add_segment(start, end):
                segment_data = {
                    "obs": self.obses[start:end],
                    "act": self.actions[start:end],
                    "true_rew": self.true_rewards[start:end],
                    "pred_rew": self.pred_rewards[start:end],
                    "done": self.terminateds[start:end],
                    "truncated": self.truncateds[start:end],
                    "collision": self.collisions[start:end],
                    "success": self.successes[start:end],
                    "image": self.images[start:end]
                }

                # Convert to JSON-safe format
                segment_data = self.format_json(segment_data)

                # Append the segment to the jsonlines file
                with jsonlines.open(self.file_db, mode='a') as writer:
                    writer.write(segment_data)
                print(f"Saved segment from index {start} to {end} to {self.file_db}")

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
                
                # Add the extra segment to the reward model
                extra_segment = {
                    "obs": combined_segment(self.obses),
                    "act": combined_segment(self.actions),
                    "true_rew": combined_segment(self.true_rewards),
                    "pred_rew": combined_segment(self.pred_rewards),
                    "done": combined_segment(self.terminateds),
                    "truncated": combined_segment(self.truncateds),
                    "collision": combined_segment(self.collisions),
                    "success": combined_segment(self.successes),
                    "image": combined_segment(self.images)
                }

                # Convert to JSON-safe format
                extra_segment = self.format_json(extra_segment)

                # Store the extra segment in the jsonlines file.
                with jsonlines.open(self.file_db, mode='a') as writer:
                    writer.write(extra_segment)
                print(f"Saved extra segment (combined from indices {cutoff} to {total}) to {self.file_db}")

            # Clear the temporary buffers.
            self.obses = []
            self.actions = []
            self.true_rewards = []
            self.pred_rewards = []
            self.terminateds = []
            self.truncateds = []
            self.collisions = []
            self.successes = []
            self.images = []

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
            self.n_turns += 1
            self.env.step()
        # Turn right
        elif action == 2:  
            self.robot.TurnRight(90, 1.5)
            print("Robot turning right")
            self.env.step(225)
            self.n_turns += 1
            self.env.step()
        else:
            pass
        
    
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        
    def _get_observation(self):
        robot_position = np.array(self.robot.data.get("position"), dtype=np.float32)
        robot_rotation = np.array(self.robot.data.get("rotation"), dtype=np.float32)
        
        robotX = robot_position[0]
        robotZ = robot_position[2]
        spongeX = self.sponge_position[0]
        spongeZ = self.sponge_position[2]
        
        robotdx = robotX - spongeX
        robotdz = robotZ - spongeZ
        robotRy = robot_rotation[1]
        # Use minimum distance for safe obstacle avoidance
        robotd2bedx = min(np.abs(robotX-self.bedXL), np.abs(robotX-self.bedXR))
        robotd2bedz = min(np.abs(robotZ-self.bedZT), np.abs(robotZ-self.bedZB))
        robotd2drawerx = min(np.abs(robotX-self.drawerXL), np.abs(robotX-self.drawerXR))
        robotd2drawerz = min(np.abs(robotZ-self.drawerZT), np.abs(robotZ-self.drawerZB))
            
        # return observation    
        return [robotdx, robotdz, robotRy, robotd2bedx, robotd2bedz, robotd2drawerx, robotd2drawerz]
        
    
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
        robot_pos = self.robot.data.get("position")
        robot_rot = self.robot.data.get("rotation")
        self.env.step()
        
        # Reward shaping.
        # Positive reward if the robot moves closer to the goal
        # Penalise timesteps to encourage efficiency
        prev_dist = self.prev_distance  
        curr_dist = np.linalg.norm(np.array(self.sponge_position) - np.array(robot_pos))
        dist_reward = (prev_dist - curr_dist) * 10
        time_reward = -0.5 # small penalty to discourage non-progress
        reward = dist_reward + time_reward
        self.prev_distance = curr_dist
        
        x_low = -2.0
        x_high = 2.0
        z_low = -1.0
        z_high = 3.0

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
            
        # Penalise excessive turns
        if self.n_turns > 2:
            reward = reward - 2  
            
        # Check if the robot is in the goal area
        full_goal, partial_goal = self.is_in_goal_area(robot_pos, robot_rot)
        if full_goal:
            print("------ robot is in goal area  ------")
            reward = reward + 30
            self.n_success += 1
            is_success = True
            is_done = True
            
        # Truncate after 40 steps per episode (max_episode length is 12)
        if self.n_eps_steps >= 40:
            truncated = True
            reward += -1 * np.linalg.norm(np.array(self.sponge_position) - np.array(robot_pos))
            
        self.returns += reward
        
        print({
            "num_steps": self.n_steps,
            "num_episodes": self.n_episodes,
            "num_episodes_steps": self.n_eps_steps,
            "num_episodes_turns": self.n_turns,
            "true reward": reward,
            "pred_reward" : pred_reward,
            "done": is_done,
            "truncated": truncated,
            "n_collisions": len(collision),
            "n_successes": self.n_success,
            "returns": self.returns,
            "counter": self.counter
        })  
       
    #    FOR EVALUATION OF RESULTS
        with jsonlines.open("ablation/llm/compare_enhanced.jsonl", mode="a") as writer:
            writer.write({
                "num_steps": self.n_steps,
                "num_episodes": self.n_episodes,
                "num_episodes_steps": self.n_eps_steps,
                "true_reward": reward,
                "pred_reward": pred_reward,
                "is_done": is_done,
                "truncated": truncated,
                "collision": collision,
                "success": is_success,
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
    
    
    
class IntrinsicRewardWrapper(VecEnvWrapper):
    def __init__(self, venv, k=2):
        super().__init__(venv)

    def reset(self):
        result = self.venv.reset()
        # If it returns (obs, info), grab only obs:
        obs = result[0] if isinstance(result, tuple) else result
        return obs

    def step_async(self, actions):
        return self.venv.step_async(actions)

    def step_wait(self):
        # Call underlying env's step_wait or step
        if hasattr(self.venv, "step_wait"):
            result = self.venv.step_wait()
        else:
            result = self.venv.step(None)
        # Unpack
        if len(result) == 5:
            obs, rewards, terminateds, truncateds, infos = result
            dones = [t or tr for t, tr in zip(terminateds, truncateds)]
        elif len(result) == 4:
            obs, rewards, dones, infos = result
        else:
            raise ValueError(f"Unexpected number of values returned by step_wait: {len(result)}")
        # Return VecEnv-like 4-tuple for off-policy algos: obs, rewards, dones, infos
        print(f"rewards type: type{rewards}")
        print(rewards)
        return obs, rewards, dones, infos

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()


# def make_env(port):
#     def _init():
#         env = ReachSponge(use_graphics=False, port=port)
#         return env
#     return _init           


# if __name__ == "__main__":
#     # Specify the ports for each parallel environment
#     manual_ports = [5000]

#     # Create a list of environment constructors
#     env_fns = [make_env(port) for port in manual_ports]
#     env = SubprocVecEnv(env_fns)

#     # Directory holding trained models
#     models_dir = "enhanced_rl/model/dqn"
#     os.makedirs(models_dir, exist_ok=True)

#     # Which checkpoints to evaluate
#     model_nums = [64000]  # e.g. [start, mid, finish]

#     # Number of evaluation episodes
#     num_episodes = 16

#     for model_num in model_nums:
#         # Load the trained DQN model
#         model_path = os.path.join(models_dir, f"enhanced_reach_sponge_dqn{model_num}")
#         model = DQN.load(model_path)
#         print(f"Loaded model checkpoint: {model_num}")

#         # Reset the vectorized env
#         obs = env.reset()

#         # Run multiple episodes
#         for ep in range(1, num_episodes + 1):
#             # Track done flags and rewards per sub-env
#             done = np.array([False] * env.num_envs)
#             total_rewards = np.zeros(env.num_envs)

#             # Step until all sub-envs signal done
#             while not np.all(done):
#                 action, _ = model.predict(obs, deterministic=True)
#                 obs, rewards, done, infos = env.step(action)
#                 total_rewards += rewards

#             print(f"Episode {ep} rewards per env: {total_rewards}")

#         print("Evaluation for this model complete.\n")

#     env.close()




              
#  LLM LABEL AND MODEL ACCURACY                  
if __name__ == "__main__":
    def make_env(port):
        return ReachSponge(use_graphics=False, port=port)

    manual_ports = [3000, 3001, 3002, 3003, 3004, 3005, 3006, 3007, 3008, 3009, 3010, 3011, 3012, 3013, 3014, 3015]
    # manual_ports = [3000, 3001, 3002, 3003, 3004, 3005, 3006, 3007]
    env_fns = [lambda port=port: make_env(port) for port in manual_ports]
    vec_env = SubprocVecEnv(env_fns)
    wrapped_env = IntrinsicRewardWrapper(vec_env)


    # Directories for models, logs, and buffer
    log_dir = "enhanced_rl/logs/dqn"
    model_dir = "enhanced_rl/model/dqn"
    buffer_dir = "enhanced_rl/replay_buffer/dqn"

    # Check for existing saved models (checkpoints)
    latest_checkpoint = None
    latest_timestep = 0
    checkpoint_files = glob.glob(os.path.join(model_dir, "enhanced_reach_sponge_dqn*.zip"))

    for f in checkpoint_files:
        basename = os.path.basename(f)
        try:
            timestep = int(basename.replace("enhanced_reach_sponge_dqn", "").replace(".zip", ""))
            if timestep > latest_timestep:
                latest_timestep = timestep
                latest_checkpoint = f
        except ValueError:
            continue
        
    # Load or initialise the model
    if latest_checkpoint is not None:
        print(f"Resuming model training from checkpoint: {latest_checkpoint}")
        model = DQN.load(latest_checkpoint, env=wrapped_env, tensorboard_log=log_dir)
        buffer_path = os.path.join(buffer_dir, f"replay_buffer_{latest_timestep}.npz")
        if os.path.exists(buffer_path):
            print(f"Loading replay buffer from: {buffer_path}")
            model.replay_buffer.load(buffer_path)
            model._setup_model()  # Ensure the model is re-bound properly
            print(f"RB size: {model.replay_buffer.pos}")
        else:
            raise FileNotFoundError("Replay buffer not found. Cannot resume training with empty buffer.")
    else:
        print("No checkpoint found. Initialising a new model.")
        model = DQN("MlpPolicy", env=wrapped_env, verbose=1, tensorboard_log=log_dir,
                    replay_buffer_class=DQNPBRLReplayBuffer, buffer_size=400000)

    # Training parameters
    TOTAL_TIMESTEPS = 252000
    SAVE_INTERVAL = 10000
    REMAINING_TIMESTEPS = TOTAL_TIMESTEPS - latest_timestep

    for i in range(latest_timestep, latest_timestep + REMAINING_TIMESTEPS, SAVE_INTERVAL):
        model.learn(total_timesteps=SAVE_INTERVAL,
                    reset_num_timesteps=False,
                    tb_log_name="dqn")

        cumulative_timestep = i + SAVE_INTERVAL
        save_path = os.path.join(model_dir, f"enhanced_reach_sponge_dqn{cumulative_timestep}.zip")
        buffer_path = os.path.join(buffer_dir, f"replay_buffer_{cumulative_timestep}.npz")

        model.save(save_path)
        # train, relabel experience and load the reward model in the replay buffer's save method'
        model.replay_buffer.save(buffer_path, source="enhanced")
        # model.replay_buffer.save_to_json("enhanced")
        print(f"Model and replay buffer saved at timestep {cumulative_timestep}")

        if cumulative_timestep == 64000:
            break

    # Now safe to close
    wrapped_env.close()