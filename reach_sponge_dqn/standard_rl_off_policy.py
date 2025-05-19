# Detailed docstrings are generated using ChatGPT and cross-checked for accuracy
# Some code from : https://github.com/empriselab/RCareWorld/blob/phy-robo-care/pyrcareworld/pyrcareworld/demo/examples/example_rl.py  

import numpy as np
import os
import glob
import random
from generate_poses import GeneratePoses

from perception import Perception  # For ablation later

import torch
import jsonlines

from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnvWrapper

from stable_baselines3 import DQN
from CustomRB import DQNPBRLReplayBuffer


try:
    import gymnasium as gym
except ImportError:
    print("This feature requires gymnasium, please install with `pip install gymnasium`")
    raise
from gymnasium import spaces

from pyrcareworld.envs.bathing_env import BathingEnv

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

# Make sure these directories exist 
os.makedirs("standard_rl/logs/dqn", exist_ok=True)
os.makedirs("standard_rl/models/dqn", exist_ok=True)
os.makedirs("standard_rl/replay_buffers/dqn", exist_ok=True)

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
        
        self.counter = 0
        
        # setup directories
        self.log_dir = "standard_rl/logs/dqn"
        self.model_dir = "standard_rl/models/dqn"
        self.replay_buffer_dir = "standard_rl/replay_buffers/dqn"

        self.robot = self.env.get_robot()  # Get the robot instance in the current environment
        self.gripper = self.env.get_gripper()   #  Get the gripper too
        self.sponge = self.env.get_sponge()   #  Get the sponge
        self.env.step()
        
        self.n_steps = 0 # Initialise an n_steps variable to keep track of number of steps taken per episode. 
        self.n_episodes = 0 # Initialise an n_episodes variable to keep track of the total number of episodes
        self.n_eps_steps = 0  # Initialise an n_eps_steps to keep track of the total number of steps taken in the current episode
        self.n_success = 0   # Track the number of successes
        self.n_turns = 0   # Track number of turns in current episode to discourage excessive turning
        self.turned = False
        
        self.action_space = spaces.Discrete(3, start=0) #  define the action space - Discrete as recommended by the rcareworld authors
              
        # define the observation space.
        low = np.concatenate([
            np.array([-5.0, -5.0, -360, -5.0, -5.0, -5.0, -5.0]) 
        ])  
        high = np.concatenate([
            np.array([5.0, 5.0, 360, 5.0, 5.0, 5.0, 5.0])             
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
        
        self.sponge_position = np.array(self.sponge.data.get("position"), dtype=np.float32)  # sponge_location is the goal position
        self.env.step()  
        
        self.poses = GeneratePoses().test_poses
        #  Randomise Robot's initial position and raise the gripper on initialisation
        # For position
        pose = self.np_random.choice(self.poses)
        x, y, z = pose[0][0], pose[0][1], pose[0][2]
        self.robot.SetPosition([x, y, z])
        # For rotation
        xrot, yrot, zrot = pose[1][0], pose[1][1], pose[1][2]
        # Set the robot's rotations
        self.robot.SetRotation([xrot, yrot, zrot])  
        self.env.step(50)
        print(f"Position --- {[x, y, z]}")
        print(f"rotation --- {[xrot, yrot, zrot]}")
        
        self.returns = 0
        # Reward shaping - for evaluation
        self.prev_distance = np.linalg.norm(np.array(self.sponge.data.get("position")) - np.array(self.robot.data.get("position")))
        self.env.step()
         
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
        self.env.step()
         
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
        self._perform_action(action)  # Apply the selected action
        observation = self._get_observation()  # Get the updated observation

        # Compute: true reward, pretrain reward, and others
        reward, terminated, truncated, n_collision, is_success = self._compute_reward()
                    
        if terminated or truncated:
            self.n_episodes += 1
            self.n_eps_steps = 0
            self.n_turns = 0
        
        info = {}

        # Return the step tuple: observation, predicted_reward, done, and additional info for policy update
        return observation, reward, terminated, truncated, info
        
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    def render(self):
        pass
    
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    def close(self):
        self.env.close()
         
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    def _perform_action(self, action):
        self.turned = False
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
            self.turned = True
            self.env.step()
        # Turn right
        elif action == 2:  
            self.robot.TurnRight(90, 1.5)
            print("Robot turning right")
            self.env.step(225)
            self.n_turns += 1
            self.turned = True
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
    
    def _compute_reward(self):
        # Check for collisions
        self.env.GetCurrentCollisionPairs()
        self.env.step()
        collision = self.env.data["collision_pairs"]
        
        # Get reward, whether the episode is over, and is_success if the episode was successful
        reward, is_done, truncated, is_success = self._get_reward(collision)
        return reward, is_done, truncated, len(collision), is_success
    
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def _get_reward(self, collision):
        robot_pos = self.robot.data.get("position")
        robot_rot = self.robot.data.get("rotation")
        self.env.step()
        
        # Reward shaping: Positive reward if the robot moves closer to the goal.  Penalise timesteps to encourage efficiency.
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
        if self.n_turns > 2 and self.turned:
            reward = reward + (-1 * min(10, self.n_turns-2))
            
        # Check if the robot is in the goal area
        goal = self.is_in_goal_area(robot_pos)
        if goal:
            print("------ robot is in goal area  ------")
            reward = reward + 30
            self.n_success += 1
            is_success = True
            is_done = True
            
        # Truncate after 40 steps per episode (max_episode length is 12)
        if self.n_eps_steps >= 30:
            truncated = True
            reward += -0.5 * np.linalg.norm(np.array(self.sponge_position) - np.array(robot_pos))
            
        self.returns += reward
        
        print({
            "num_steps": self.n_steps,
            "num_episodes": self.n_episodes,
            "num_episodes_steps": self.n_eps_steps,
            "num_episodes_turns": self.n_turns,
            "true reward": reward,
            # "pred_reward" : pred_reward,
            "done": is_done,
            "truncated": truncated,
            "n_collisions": len(collision),
            "n_successes": self.n_success,
            "returns": self.returns,
            "counter": self.counter
        })  
       
    #    FOR EVALUATION OF RESULTS
        with jsonlines.open("ablation/llm/compare_standard.jsonl", mode="a") as writer:
            writer.write({
                "num_steps": self.n_steps,
                "num_episodes": self.n_episodes,
                "num_episodes_steps": self.n_eps_steps,
                "true_reward": reward,
                # "pred_reward": pred_reward,
                "is_done": is_done,
                "truncated": truncated,
                "collision": collision,
                "success": is_success,
            })
        print(f"Results saved to file.")
        self.turned = False
        return (reward, is_done, truncated, is_success)
    
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # Check if the robot is in an easy grasp position
    def is_in_goal_area(self, robot_pos):
        # Define the goal area bounds
        xmax, xmin = -0.050, -0.258
        # ymin, ymax = 0, 0  # ignore 
        zmax, zmin = 1.725, 1.400
        # Extract robot position 
        x, z = robot_pos[0], robot_pos[2]
        # Check position constraints
        valid_pos = xmin <= x <= xmax and zmin <= z <= zmax
        
        return (valid_pos)
    
    
class IntrinsicRewardWrapper(VecEnvWrapper):
    def __init__(self, venv):
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
        with jsonlines.open("standard_rl/logs/dqn/wrapper_timesteps.jsonl", mode="a") as writer:
            writer.write({
                "reward": rewards.tolist(),
                "observation": obs.tolist(),
                "is_done": dones.tolist(),
            })
        return obs, rewards, dones, infos

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()
        


def make_env(port):
    def _init():
        env = ReachSponge(use_graphics=False, port=port)
        return env
    return _init       


if __name__ == "__main__":
    # Specify the ports for each parallel environment
    manual_ports = [5000]

    # Create a list of environment constructors
    env_fns = [make_env(port) for port in manual_ports]
    env = SubprocVecEnv(env_fns)

    # Directory holding trained models
    models_dir = "standard_rl/models/dqn"
    os.makedirs(models_dir, exist_ok=True)

    # Which checkpoints to evaluate
    model_nums = [32000]  # e.g. [start, mid, finish]

    # Number of evaluation episodes
    num_episodes = 16

    for model_num in model_nums:
        # Load the trained DQN model
        model_path = os.path.join(models_dir, f"standard_reach_sponge_dqn{model_num}")
        model = DQN.load(model_path)
        print(f"Loaded model checkpoint: {model_num}")

        # Reset the vectorized env
        obs = env.reset()

        # Run multiple episodes
        for ep in range(1, num_episodes + 1):
            # Track done flags and rewards per sub-env
            done = np.array([False] * env.num_envs)
            total_rewards = np.zeros(env.num_envs)

            # Step until all sub-envs signal done
            while not np.all(done):
                action, _ = model.predict(obs, deterministic=True)
                obs, rewards, done, infos = env.step(action)
                total_rewards += rewards

            print(f"Episode {ep} rewards per env: {total_rewards}")

        print("Evaluation for this model complete.\n")

    env.close()
                       
                   
# if __name__ == "__main__":
#     def make_env(port):
#         return ReachSponge(use_graphics=False, port=port)

#     manual_ports = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015]
#     env_fns = [lambda port=port: make_env(port) for port in manual_ports]
#     vec_env = SubprocVecEnv(env_fns)
#     wrapped_env = IntrinsicRewardWrapper(vec_env)

#     # Directories for models, logs, and buffer
#     log_dir = "standard_rl/logs/dqn"
#     model_dir = "standard_rl/models/dqn"
#     buffer_dir = "standard_rl/replay_buffers/dqn"

#     # Check for existing saved models (checkpoints)
#     latest_checkpoint = None
#     latest_timestep = 0
#     checkpoint_files = glob.glob(os.path.join(model_dir, "standard_reach_sponge_dqn*.zip"))

#     for f in checkpoint_files:
#         basename = os.path.basename(f)
#         try:
#             timestep = int(basename.replace("standard_reach_sponge_dqn", "").replace(".zip", ""))
#             if timestep > latest_timestep:
#                 latest_timestep = timestep
#                 latest_checkpoint = f
#         except ValueError:
#             continue

#     # Load or initialise the model
#     if latest_checkpoint is not None:
#         print(f"Resuming model training from checkpoint: {latest_checkpoint}")
#         model = DQN.load(latest_checkpoint, env=wrapped_env, tensorboard_log=log_dir)
#         buffer_path = os.path.join(buffer_dir, f"replay_buffer_{latest_timestep}.npz")
#         if os.path.exists(buffer_path):
#             print(f"Loading replay buffer from: {buffer_path}")
#             model.replay_buffer.load(buffer_path)
#             model._setup_model()  # Ensure the model is re-bound properly
#             print(f"Single env RB size: {model.replay_buffer.pos}")
#         else:
#             raise FileNotFoundError("Replay buffer not found. Cannot resume training with empty buffer.")
#     else:
#         print("No checkpoint found. Initialising a new model.")
#         model = DQN("MlpPolicy", env=wrapped_env, verbose=1, tensorboard_log=log_dir,
#                     replay_buffer_class=DQNPBRLReplayBuffer, buffer_size=400000)

#     # Training parameters
#     TOTAL_TIMESTEPS = 356000
#     SAVE_INTERVAL = 20000
#     REMAINING_TIMESTEPS = TOTAL_TIMESTEPS - latest_timestep

#     for i in range(latest_timestep, latest_timestep + REMAINING_TIMESTEPS, SAVE_INTERVAL):
#         model.learn(total_timesteps=SAVE_INTERVAL,
#                     reset_num_timesteps=False,
#                     tb_log_name="dqn")

#         cumulative_timestep = i + SAVE_INTERVAL
#         save_path = os.path.join(model_dir, f"standard_reach_sponge_dqn{cumulative_timestep}.zip")
#         buffer_path = os.path.join(buffer_dir, f"replay_buffer_{cumulative_timestep}.npz")

#         model.save(save_path)
#         model.replay_buffer.save(buffer_path)
#         model.replay_buffer.save_to_json("standard")
#         print(f"Model and replay buffer saved at timestep {cumulative_timestep}")

#     wrapped_env.close()