# Detailed docstrings are generated using ChatGPT and cross-checked for accuracy
# Some code from : https://github.com/empriselab/RCareWorld/blob/phy-robo-care/pyrcareworld/pyrcareworld/demo/examples/example_rl.py  

import numpy as np
import os
import glob
import random
from generate_poses import GeneratePoses

from perception import Perception

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

# Make sure these directories exist - logs dir, pretrain rb, pretrain rm, pretrain model
os.makedirs("pretrain/logs/enhanced_rl/dqn", exist_ok=True)
os.makedirs("pretrain/reward_model/enhanced_rl/dqn", exist_ok=True)
os.makedirs("pretrain/model/enhanced_rl/dqn", exist_ok=True)
os.makedirs("pretrain/replay_buffer/enhanced_rl/dqn", exist_ok=True)

def compute_state_entropy_batch(obs, full_obs, k):
        batch_size = 500
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        full_obs_tensor = torch.tensor(full_obs, dtype=torch.float32)
        with torch.no_grad():
            dists = []
            for idx in range(len(full_obs_tensor) // batch_size + 1):
                start = idx * batch_size
                end   = (idx + 1) * batch_size
                dist = torch.norm(
                    obs_tensor[:, None, :] - full_obs_tensor[None, start:end, :],
                    dim=-1, p=2
                )
                dists.append(dist)
            dists = torch.cat(dists, dim=1)
            knn_dists = torch.kthvalue(dists, k=k + 1, dim=1).values
            state_entropy = knn_dists

        return state_entropy.squeeze(-1)



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
        
        # setup directories
        self.log_dir = "pretrain/logs/enhanced_rl/dqn"
        self.model_dir = "pretrain/model/enhanced_rl/dqn"
        self.reward_model_dir = "pretrain/reward_model/enhanced_rl/dqn"
        self.replay_buffer_dir = "pretrain/replay_buffer/enhanced_rl/dqn"

        self.robot = self.env.get_robot()  # Get the robot instance in the current environment
        self.gripper = self.env.get_gripper()   #  Get the gripper too
        self.sponge = self.env.get_sponge()   #  Get the sponge
        self.env.step()
        
        self.n_steps = 0 # Initialise an n_steps variable to keep track of number of steps taken per episode. 
        self.n_episodes = 0 # Initialise an n_episodes variable to keep track of the total number of episodes
        self.n_eps_steps = 0  # Initialise an n_eps_steps to keep track of the total number of steps taken in the current episode
        self.n_success = 0   # Track the number of successes
        self.n_turns = 0   # Track number of turns in current episode to discourage excessive turning
        
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
        
        self.poses = GeneratePoses().train_poses
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
        if self.n_turns > 2:
            reward = reward - 2  
            
        # Check if the robot is in the goal area
        goal = self.is_in_goal_area(robot_pos)
        if goal:
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
            "done": is_done,
            "truncated": truncated,
            "n_collisions": len(collision),
            "n_successes": self.n_success,
            "returns": self.returns
        })  
       
        with jsonlines.open("pretrain/logs/enhanced_rl/dqn/timesteps.jsonl", mode="a") as writer:
            writer.write({
                "true_reward": reward,
                "is_done": is_done,
                "truncated": truncated,
                "collision": len(collision) > 0,
                "success": is_success
            })
        # print(f"Results saved to file.")
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
    def __init__(self, venv, k=2):
        super().__init__(venv)
        self.k = k
        self.full_obs = []       # one list for all sub-envs

    def reset(self):
        result = self.venv.reset()
        # If it returns (obs, info), grab only obs:
        obs = result[0] if isinstance(result, tuple) else result
        self.full_obs.extend(obs.tolist())
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
            obs, _, terminateds, truncateds, infos = result
            dones = [t or tr for t, tr in zip(terminateds, truncateds)]
        elif len(result) == 4:
            obs, _, dones, infos = result
        else:
            raise ValueError(f"Unexpected number of values returned by step_wait: {len(result)}")
        # Track obs history
        self.full_obs.extend(obs.tolist())
        # Compute intrinsic reward only
        r_int = compute_state_entropy_batch(obs, self.full_obs, self.k)
        with jsonlines.open("pretrain/logs/enhanced_rl/dqn/wrapper_timesteps.jsonl", mode="a") as writer:
            writer.write({
                "Intrinsic_reward": r_int.cpu().numpy().tolist(),
                "observation": obs.tolist(),
                "is_done": dones.tolist(),
            })
        # Return VecEnv-like 4-tuple for off-policy algos: obs, rewards, dones, infos
        return obs, r_int.cpu().numpy(), dones, infos

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()
                      
                   
if __name__ == "__main__":
    def make_env(port):
        return ReachSponge(use_graphics=False, port=port)

    manual_ports = [1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015]
    env_fns = [lambda port=port: make_env(port) for port in manual_ports]
    vec_env = SubprocVecEnv(env_fns)
    wrapped_env = IntrinsicRewardWrapper(vec_env, k=2)

    # Directories for models, logs, and buffer
    log_dir = "pretrain/logs/enhanced_rl/dqn"
    model_dir = "pretrain/model/enhanced_rl/dqn"
    buffer_dir = "pretrain/replay_buffer/enhanced_rl/dqn"

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
    TOTAL_TIMESTEPS = 32000
    SAVE_INTERVAL = 8000
    REMAINING_TIMESTEPS = TOTAL_TIMESTEPS - latest_timestep

    for i in range(latest_timestep, latest_timestep + REMAINING_TIMESTEPS, SAVE_INTERVAL):
        model.learn(total_timesteps=SAVE_INTERVAL,
                    reset_num_timesteps=False,
                    tb_log_name="dqn")

        cumulative_timestep = i + SAVE_INTERVAL
        save_path = os.path.join(model_dir, f"enhanced_reach_sponge_dqn{cumulative_timestep}.zip")
        buffer_path = os.path.join(buffer_dir, f"replay_buffer_{cumulative_timestep}.npz")

        model.save(save_path)
        model.replay_buffer.save(buffer_path)
        model.replay_buffer.save_to_json()
        print(f"Model and replay buffer saved at timestep {cumulative_timestep}")

    wrapped_env.close()