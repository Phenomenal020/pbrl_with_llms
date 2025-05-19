import os
import glob

import jsonlines
import numpy as np
import cv2
import random

import time
from datetime import datetime

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnvWrapper
from stable_baselines3.common.buffers import ReplayBuffer

from pyrcareworld.envs.bathing_env import BathingEnv
import pyrcareworld.attributes.camera_attr as attr
from perception_test3 import Perception
from pyrcareworld.attributes import PersonRandomizerAttr

from CustomRB import DQNPBRLReplayBuffer
from reward_model_image import RewardModel



import torch

try:
    import gymnasium as gym
except ImportError:
    print("This feature requires gymnasium, please install with `pip install gymnasium`")
    raise
from gymnasium import spaces

os.makedirs("enhanced_rl/logs/dqn", exist_ok=True)
os.makedirs("enhanced_rl/model/dqn", exist_ok=True)
os.makedirs("enhanced_rl/reward_model/dqn", exist_ok=True)
os.makedirs("enhanced_rl/replay_buffer/dqn", exist_ok=True)

os.makedirs("enhanced_rl/observations", exist_ok=True)

class ReachMannequinEnv(gym.Env):
    # metadata = {'render.modes': ['human']}

    def __init__(self, port=None, use_graphics=True, dev=False):
        super(ReachMannequinEnv, self).__init__()
        # save the args so reset() can re-create from them
        self.port = port
        self.use_graphics = use_graphics
        self.dev = dev

        # call helper to spawn Unity, set up everything
        self._make_env()

        # episode counters
        self.step_count = 0
        self.zstep_count = 0
        self.n_success = 0
        self.num_episodes = 0
        self.max_steps = 15
        self.total_steps = 0
        self.imgdir = "enhanced_rl/observations"
        # self.imgdir2 = "enhanced_rl/observations2"
        self.env.step()
        
        person_randomizer = PersonRandomizerAttr(self.env, 573920)
        person_randomizer.SetSeed(42)
        self.env.step()
        
        # Initialise reward model to get predicted rewards
        self.load_step = 4200
        self.reward_model = RewardModel(ds=7, da=3)
        loaded_files = self.reward_model.load(model_dir="enhanced_rl/reward_model/dqn", load_step=4200)
        if not loaded_files:
            raise Exception("No reward model found. A new model will be trained.")
        self.env.step()
        
        # temporary buffers to hold data before adding to the reward model
        self.obses = []
        self.imageobses = []
        self.actions = []
        self.true_rewards = []
        self.pred_rewards = []
        self.terminateds = []
        self.truncateds = []
        self.goals = []
        self.env.step()
        
        
        # Directory for data storage
        os.makedirs("enhanced_rl/db/trajectories", exist_ok=True)
        self.file_db = f"enhanced_rl/db/trajectories/enhanced_reach_sponge{port}.jsonl"
        self.env.step()
        

    def _make_env(self):

        self.env = BathingEnv(
            graphics=self.use_graphics,
            port=self.port,
        ) if not self.dev else BathingEnv(
            graphics=self.use_graphics,
            executable_file="@editor",
            port=self.port
        )
        
        self.person_midpoint = [0.055898, 0.84, 0.003035]  # initialise with a confirmed midpoint
                
        # Setup a camera to capture the entire scene
        self.env.scene_camera = self.env.InstanceObject(name="Camera", id=123456, attr_type=attr.CameraAttr) 
        # Set position and orientation for the scene camera
        self.env.scene_camera.SetTransform(position=[0, 4.2, 0.6], rotation=[90, 0, 0.0]) 
        self.env.scene_camera.Get3DBBox()
        self.env.step()
        for i in self.env.scene_camera.data["3d_bounding_box"]:
            if i == 573920:
                self.person_midpoint = self.env.scene_camera.data["3d_bounding_box"][i][0]
                print(f"Person midpoint: {self.person_midpoint}")
        self.env.step()
        
        # Setup a camera to get manikin data
        self.env.manikin_camera = self.env.InstanceObject(name="Camera", id=333333, attr_type=attr.CameraAttr)
        self.env.manikin_camera.SetTransform(position=[0.0, 2.1, 0.15], rotation=[90, 0, 0])
        self.env.step() 
        
        self.obscamera = self.env.InstanceObject(name="Camera", id=666666,attr_type=attr.CameraAttr)
        self.obscamera.SetTransform(position=[-0.4, 1.36, 0.36], rotation=[40, 105, 0])
        self.env.step()
        
        # self.obscamera2 = self.env.InstanceObject(name="Camera", id=777777,attr_type=attr.CameraAttr)
        # self.obscamera2.SetTransform(position=[1, 2.0, 0.0], rotation=[45, 270, 0])
        # self.env.step()
       
        # 3) perception → trajectories
        p = Perception(self.env, port=self.port)
        p.get_landmarks()
        p.generate_trajectory()
        self.trajectories     = p.trajectories
        self.manikin_indices  = [4, 5, 6]
        self.current_probe    = random.randint(0, len(self.manikin_indices)-1)
        print(f"Current probe: {self.current_probe}")
        
        self.robot  = self.env.get_robot()
        self.gripper = self.env.get_gripper()
        self.sponge = self.env.get_sponge()
        self.initial_sponge_pos = self.sponge.data.get("position")
        self.force_thresh = 6.0

        # 5) define gym spaces
        self.action_space = spaces.Discrete(3)
        low  = np.array([-5, -5, -5, 0, -5, -5, -5], dtype=np.float32)
        high = np.array([ 5,  5,  5, 100,  5,  5,  5], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        

    def reset(self, seed=None, options=None):
        # increment episode counter
        self.num_episodes += 1

        # zero your per-episode counters
        self.step_count  = 0
        self.zstep_count = 0
        self.current_probe = random.randint(0, len(self.manikin_indices)-1)
        print(f"Current probe: {self.current_probe}")

        # if we’ve already run at least one episode, tear down & rebuild
        if self.num_episodes > 1:
            self.env.close()
            self._make_env()

        # your usual “first movements” (grasp, dip, etc.)
        if self.sponge.data.get("position") == self.initial_sponge_pos:
            self._grasp_sponge()
            self._dip_sponge()
            self._approach_manikin()
            self._preprobe()

        obs = self._get_observation()
        
        print("resetting ***************************************************")
        return obs, {}


    def step(self, action):
        obs = self._get_observation()
        self.prev_dist = [obs[0], obs[1], obs[2]]
        dxyz = None
        if action == 0:
            dxyz = [0, -0.02, 0]  # move down
        elif action == 1:
            dxyz = [0, 0, -0.05]  # move forward
            self.zstep_count += 1
        elif action == 2:
            dxyz = [0, 0.02, 0]  # move up
        
        # apply delta movement
        if dxyz is not None:
            cur = np.array(self.gripper.data.get('grasp_point_position'))
            targ = (cur + np.array(dxyz)).tolist()
            self.robot.IKTargetDoMove(
                position=targ, 
                duration=2,
                speed_based=False, 
                relative=False
            )
            self.robot.WaitDo()
            self.env.step(100)

        self.step_count += 1
        self.total_steps += 1
        obs = self._get_observation()
        force = float(self.sponge.GetForce()[0])
        terminated = force >= 0.05
        truncated = self.step_count >= self.max_steps
        rew_z, rew_y = 0, 0
        if self.zstep_count <= 2:
            rew_z = (self.prev_dist[2] - obs[2]) * 10
            # print(f"rew_z: {rew_z} *** action: {action}")
        else:
            rew_z = (obs[2] - self.prev_dist[2]) * 10
            # print(f"rew_z: {rew_z} *** action: {action}")
        if obs[1] > 0.84:
            rew_y = (self.prev_dist[1] - obs[1]) * 20
            # print(f"rew_y: {rew_y} *** action: {action}")
        else:
            rew_y = (obs[1] - self.prev_dist[1]) * 20
            # print(f"rew_y: {rew_y} *** action: {action}")
        reward = 20 if terminated else rew_z + rew_y 
        if reward == 20 and terminated:
            self.n_success += 1
        if terminated or truncated:
            self.num_episodes += 1
            
        # Compute predicted and true rewards based on the new obs
        predicted_reward = self.reward_model.r_hat(np.array(obs))
        predicted_reward = predicted_reward.detach().cpu().item() 
        
        image = self.obscamera.GetRGB(width=512, height=512)
        self.env.step()
        rgb = np.frombuffer(self.obscamera.data["rgb"], dtype=np.uint8)
        imageobs = cv2.imdecode(rgb, cv2.IMREAD_COLOR)
        if imageobs is None:
            print("Error: Decoded image is None.")
            return
        else:
            print("Image successfully captured. Image shape:", imageobs.shape) 
        image = cv2.cvtColor(imageobs, cv2.COLOR_RGB2BGR)
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}{self.port}.png"
        print(f"filename: {filename}")
        cv2.imwrite(os.path.join(self.imgdir, filename), image)
        
        # image2 = self.obscamera2.GetRGB(width=512, height=512)
        # self.env.step()
        # rgb2 = np.frombuffer(self.obscamera2.data["rgb"], dtype=np.uint8)
        # imageobs2 = cv2.imdecode(rgb2, cv2.IMREAD_COLOR)
        # if imageobs2 is None:
        #     print("Error: Decoded image is None.")
        #     return
        # else:
        #     print("Image successfully captured. Image shape:", imageobs2.shape)  
        # image2 = cv2.cvtColor(imageobs2, cv2.COLOR_RGB2BGR)
        # filename2 = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}{self.port}_2.png"
        # print(f"Filename2: {filename2}")
        # cv2.imwrite(os.path.join(self.imgdir2, filename2), image2)
        
        goal = None
        if self.current_probe == 0:
            goal = "arm"
        elif self.current_probe == 1:
            goal = "torso"
        elif self.current_probe == 2:
            goal = "leg"
        
        # store data temporarilly (for reward model training)
        self.obses.append(obs)
        self.imageobses.append(filename)
        # self.imageobses2.append(filename2)
        self.actions.append(action)
        self.true_rewards.append(reward)
        self.pred_rewards.append(predicted_reward)
        self.terminateds.append(terminated)
        self.truncateds.append(truncated)
        # self.collisions.append(n_collision)
        self.goals.append(goal)

        # Check if episode is done (terminated or truncated), then add data in segments of 5 to the reward model.  if episode length < 5, discard. If last segment < 5, combine with penultimate segment, then add of length 10 to the terminated step.
        self.process_episode(terminated, truncated)   
        
        print(f"Step: {self.step_count} ----- Force: {force} ---- Action: {action} ---- True_Reward: {reward} ---- Predicted_Reward: {predicted_reward} ---- Terminated: {terminated} ---- Truncated: {truncated} ---- num_episodes: {self.num_episodes} ---- n_success: {self.n_success} ---- total_steps: {self.total_steps}")
        
        
        return obs, predicted_reward, terminated, truncated, {}
    
    
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
            if total < 5:  # Discard
                # Clear the temporary buffer and return.
                self.obses = []
                self.imageobses = []
                # self.imageobses2 = []
                self.actions = []
                self.true_rewards = []
                self.pred_rewards = []
                self.terminateds = []
                self.truncateds = []
                # self.collisions = []
                self.goals = []
                return

            # Else, 1. get the remainder of the total number of observations in the episode
            remainder = total % 5

            # 2. Helper function to add a segment (size 5) for all data fields.
            def add_segment(start, end):
                segment_data = {
                    "obs": self.obses[start:end],
                    "imageobs": self.imageobses[start:end],
                    # "imageobs2": self.imageobses2[start:end],
                    "act": self.actions[start:end],
                    "true_rew": self.true_rewards[start:end],
                    "pred_rew": self.pred_rewards[start:end],
                    "done": self.terminateds[start:end],
                    "truncated": self.truncateds[start:end],
                    # "collision": self.collisions[start:end],
                    "goal": self.goals[start:end]
                }

                # Convert to JSON-safe format
                segment_data = self.format_json(segment_data)

                # Append the segment to the jsonlines file
                with jsonlines.open(self.file_db, mode='a') as writer:
                    writer.write(segment_data)
                print(f"Saved segment from index {start} to {end} to {self.file_db}")

            #3. If the data divides evenly into batches of 5, process them normally.
            if remainder == 0:
                for i in range(0, total, 5):
                    add_segment(i, i+5)
            else: # Otherwise,
                #4. If there is an incomplete last segment.
                # Process all complete segments up to the penultimate one.
                # The penultimate full segment starts at index: cutoff = total - remainder - 10.
                cutoff = total - remainder - 5 # (eg, if total = 87, then remainder = 7 and cutoff = 70)
                # Add data normally up to cutoff
                for i in range(0, cutoff, 5):
                    add_segment(i, i+5)

                # Instead of adding the penultimate and the last incomplete segments separately, combine them and adjust by slicing backwards twice.
                def combined_segment(data):
                    # Concatenate penultimate full segment and the incomplete segment.
                    penultimate = data[cutoff:cutoff+5]
                    incomplete = data[cutoff+5: total]
                    combined = penultimate + incomplete  
                    # Reverse the combined list and take the first 5 elements.
                    reversed_list = combined[::-1][:5]
                    # Reverse again to maintain the original order.
                    final_list = reversed_list[::-1]
                    return final_list
                
                # Add the extra segment to the reward model
                extra_segment = {
                    "obs": combined_segment(self.obses),
                    "imageobs": combined_segment(self.imageobses),
                    # "imageobs2": combined_segment(self.imageobses2),
                    "act": combined_segment(self.actions),
                    "true_rew": combined_segment(self.true_rewards),
                    "pred_rew": combined_segment(self.pred_rewards),
                    "done": combined_segment(self.terminateds),
                    "truncated": combined_segment(self.truncateds),
                    # "collision": combined_segment(self.collisions),
                    "goal": combined_segment(self.goals),
                }

                # Convert to JSON-safe format
                extra_segment = self.format_json(extra_segment)

                # Store the extra segment in the jsonlines file.
                with jsonlines.open(self.file_db, mode='a') as writer:
                    writer.write(extra_segment)
                print(f"Saved extra segment (combined from indices {cutoff} to {total}) to {self.file_db}")

            # Clear the temporary buffers.
            self.obses = []
            self.imageobses = []
            # self.imageobses2 = []
            self.actions = []
            self.true_rewards = []
            self.pred_rewards = []
            self.terminateds = []
            self.truncateds = []
            # self.collisions = []
            self.goals = []

        else:
            return


    def close(self):
        self.env.close()

    # helper methods
    def _get_observation(self):
        ee = self.gripper.data.get('grasp_point_position')
        force = float(self.sponge.GetForce()[0])
        pm = self.person_midpoint
        print(f"self.person_midpoint: {self.person_midpoint}")
        return [ee[0], ee[1], ee[2], force, pm[0], pm[1], pm[2]]

    def _grasp_sponge(self):
         # Raise the gripper to a safe height for easy obstacle avoidance
        gripper_pos = [self.robot.data["position"][0], self.initial_sponge_pos[1] + 0.2, self.robot.data["position"][2]]
        self.robot.IKTargetDoMove(
            position=gripper_pos,
            duration=1,
            speed_based=False,
        )
        self.robot.WaitDo()
        self.env.step(50)
        self.gripper.GripperOpen()
        self.env.step(50)
        
        # Move towards sponge
        self.robot.MoveForward(0.25, 2)
        self.env.step(100)
        self.robot.TurnLeft(90, 1.5)
        self.env.step(200)
        self.robot.MoveForward(0.2, 2)
        self.env.step(50)
        
        # Lower gripper (to grasp sponge)
        lower_position = [self.sponge.data.get("position")[0], self.initial_sponge_pos[1]+0.03, self.sponge.data.get("position")[2]]
        self.robot.IKTargetDoMove(
            position=lower_position,  
            duration=1,
            speed_based=False,
        )
        self.robot.WaitDo()
        self.env.step(50)
        self.gripper.GripperClose()
        self.env.step(50)
        
        # Raise Gripper after grasping sponge
        self.robot.IKTargetDoMove(
            position=gripper_pos,
            duration=1,
            speed_based=False,
        )
        self.robot.WaitDo()
        self.env.step(50)
        
    
    def _dip_sponge(self):
        # Move backwards
        self.robot.MoveBack(0.40, 2)
        self.env.step(50)
            
        # Lower Gripper to dip sponge
        lower_position = [self.sponge.data.get("position")[0], self.initial_sponge_pos[1]+0.03, self.sponge.data.get("position")[2]]
        self.robot.IKTargetDoMove(
            position=lower_position,  
            duration=1,
            speed_based=False,
        )
        self.robot.WaitDo()
        self.env.step(50)
        
        # Raise Gripper
        reference_gripper_pos = [self.robot.data["position"][0], self.initial_sponge_pos[1] + 0.5, self.robot.data["position"][2]]
        self.robot.IKTargetDoMove(
            position=reference_gripper_pos,
            duration=1,
            speed_based=False,
        )
        self.robot.WaitDo()
        self.env.step(50)
                 
    
    def _approach_manikin(self):
         # Turn to face Manikin
        self.robot.TurnLeft(90, 1)
        self.env.step(300)

        # Move towards Manikin
        self.robot.MoveForward(0.90, 2)
        self.env.step(200)
        
        self.robot.TurnLeft(90, 1)
        self.env.step(300)
 
        
        
    
    def _preprobe(self):

        # get the EE's current and target positions
        current = self.gripper.data.get("grasp_point_position")
        start = self.trajectories[self.manikin_indices[self.current_probe]]
        print(f"Target before probe: {start}")
        error = np.array(start) - np.array(current)
        print(f"Error before correction: {error}")
            
        # align the robot x axis (base)
        timeout = 10
        start_time = time.time()
        while abs(error[0]) > 0.05 and (time.time() - start_time) < timeout:
            if error[0] < 0:
                self.robot.MoveBack(-error[0], 1)
                self.env.step(100)
            else:
                self.robot.MoveForward(error[0], 1)
                self.env.step(100)
            error = np.array(start) - np.array(self.gripper.data.get("grasp_point_position"))
        elapsed = time.time() - start_time
        print(f"Took {elapsed:.2f}s (timeout was {timeout}s)")
        print(f"Error after x correction: {error}")
             
        # align the z axis - define a timeout (in seconds)
        timeout = 13
        start_time = time.time()
        while abs(error[2]) > 0.05 and (time.time() - start_time) < timeout:
            self.robot.IKTargetDoMove(
                position=[0, 0, error[2]],
                duration=4,
                speed_based=False,
                relative=True
            )
            self.robot.WaitDo()
            self.env.step(100)
            error = np.array(start) - np.array(self.gripper.data.get("grasp_point_position"))
        elapsed = time.time() - start_time
        print(f"Error after z correction: {error}")
        print(f"Took {elapsed:.2f}s (timeout was {timeout}s)")
               
                   
    def _move_ee(self, dxyz):
        current = np.array(self.gripper.data.get('grasp_point_position'))
        target = (current + np.array(dxyz)).tolist()
        self.robot.IKTargetDoMove(
            position=target,
            duration=2,
            speed_based=False,
            relative=False
        )
        self.robot.WaitDo()
        self.env.step(100)


class IntrinsicRewardWrapper(VecEnvWrapper):
    def __init__(self, venv):
        super().__init__(venv)

    def reset(self):
        result = self.venv.reset()
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
        with jsonlines.open("enhanced_rl/logs/dqn/wrapper_timesteps.jsonl", mode="a") as writer:
            writer.write({
                "rewards": rewards.tolist(),
                "observation": obs.tolist(),
                "is_done": dones.tolist(),
            })
        return obs, rewards.tolist(), dones, infos
    
    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()


def save_replay_buffer(buffer, path):
    np.savez_compressed(path,
        observations=buffer.observations,
        actions=buffer.actions,
        rewards=buffer.rewards,
        dones=buffer.dones,
        next_observations=buffer.next_observations,
        pos=buffer.pos,
        full=buffer.full
    )


def load_replay_buffer(buffer, path):
    data = np.load(path)
    buffer.observations = data["observations"]
    buffer.actions = data["actions"]
    buffer.rewards = data["rewards"]
    buffer.dones = data["dones"]
    buffer.next_observations = data["next_observations"]
    buffer.pos = int(data["pos"])
    buffer.full = bool(data["full"])


def make_reach_env_factory(port: int):
    def _init():
        return ReachMannequinEnv(port=port)
    return _init



if __name__ == "__main__":
    ports = [3000, 3001, 3002, 3003, 3004, 3005, 3006, 3007, 3008, 3009, 3010, 3011, 3012, 3013, 3014]
    # ports = [3000]
    env_fns = [make_reach_env_factory(p) for p in ports]
    vec_env = SubprocVecEnv(env_fns)
    wrapped_env = IntrinsicRewardWrapper(vec_env)

    # directories
    log_dir = "enhanced_rl/logs/dqn"
    model_dir = "enhanced_rl/model/dqn"
    buffer_dir = "enhanced_rl/replay_buffer/dqn"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(buffer_dir, exist_ok=True)

    # Load latest model and buffer if exists
    latest_checkpoint = None
    latest_timestep = 0
    for f in glob.glob(os.path.join(model_dir, "enhanced_rl_reach_sponge_dqn*.zip")):
        try:
            ts = int(os.path.basename(f).split("dqn")[-1].split(".zip")[0])
            if ts > latest_timestep:
                latest_timestep, latest_checkpoint = ts, f
        except ValueError:
            continue

    if latest_checkpoint:
        print(f"Resuming from {latest_checkpoint}")
        model = DQN.load(latest_checkpoint, env=wrapped_env, tensorboard_log=log_dir)
        buf_path = os.path.join(buffer_dir, f"replay_buffer_{latest_timestep}.npz")
        if os.path.exists(buf_path):
            # load_replay_buffer(model.replay_buffer, buf_path)
            model.replay_buffer.load(buf_path)
            # model._setup_model()
    else:
        print("Starting new DQN model")
        model = DQN(
            policy="MlpPolicy",
            env=wrapped_env,
            replay_buffer_class=DQNPBRLReplayBuffer,
            buffer_size=50000,
            verbose=1,
            tensorboard_log=log_dir
        )

    # Training and periodic saving
    TOTAL_TIMESTEPS = 13800
    SAVE_INTERVAL = 30
    for lts in range(latest_timestep, TOTAL_TIMESTEPS, SAVE_INTERVAL):
        model.learn(
            total_timesteps=SAVE_INTERVAL,
            reset_num_timesteps=False,
            tb_log_name="dqn"
        )
        timestep = lts + SAVE_INTERVAL
        checkpoint = os.path.join(model_dir, f"enhanced_rl_reach_sponge_dqn{timestep}.zip")
        buffer = os.path.join(buffer_dir, f"replay_buffer_{timestep}.npz")
        model.save(checkpoint)
        # save_replay_buffer(model.replay_buffer, buf)
        model.replay_buffer.save(buffer, "enhanced")
        print(f"Saved model and buffer at {timestep}")
        
        if latest_timestep == 4800:
            wrapped_env.close()