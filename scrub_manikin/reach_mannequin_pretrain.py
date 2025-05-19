import os
import glob

import jsonlines
import numpy as np
import cv2
import random
import time

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnvWrapper
from stable_baselines3.common.buffers import ReplayBuffer

from pyrcareworld.envs.bathing_env import BathingEnv
import pyrcareworld.attributes.camera_attr as attr
from perception_test3 import Perception
from pyrcareworld.attributes import PersonRandomizerAttr

from CustomRB import DQNPBRLReplayBuffer


import torch

try:
    import gymnasium as gym
except ImportError:
    print("This feature requires gymnasium, please install with `pip install gymnasium`")
    raise
from gymnasium import spaces


os.makedirs("pretrain/logs/dqn", exist_ok=True)
os.makedirs("pretrain/model/dqn", exist_ok=True)
os.makedirs("pretrain/reward_model/dqn", exist_ok=True)
os.makedirs("pretrain/replay_buffer/dqn", exist_ok=True)

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
        self.env.step()
        
        person_randomizer = PersonRandomizerAttr(self.env, 573920)
        person_randomizer.SetSeed(42)
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
        
        # 4) camera for LLM obs
        self.obscamera = self.env.InstanceObject(
            name="Camera", id=666666, attr_type=attr.CameraAttr
        )
        self.obscamera.SetTransform(
            position=[-0.1, 2, -0.1],
            rotation=[60, 45, 0.0]
        )
        self.env.step()
       
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
        high = np.array([ 5,  5,  5,100,  5,  5,  5], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        

    def reset(self, seed=None, options=None):
        # self.env.WaitSceneInit()
        # self.env.WaitLoadDone() 
        
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
        print(f"Step: {self.step_count} ----- Force: {force} ---- Action: {action} ---- Reward: {reward} ---- Terminated: {terminated} ---- Truncated: {truncated} ---- num_episodes: {self.num_episodes} ---- n_success: {self.n_success} ---- total_steps: {self.total_steps}")
        return obs, reward, terminated, truncated, {}


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
        
        print(f"robot position: {self.robot.data.get['position']}")
        print(f"robot rotation: {self.robot.data.get['rotation']}")
        
        
    
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
    def __init__(self, venv, k):
        super().__init__(venv)
        self.k = k
        self.full_obs = []

    def reset(self):
        result = self.venv.reset()
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
        with jsonlines.open("pretrain/logs/dqn/wrapper_timesteps.jsonl", mode="a") as writer:
            writer.write({
                "Intrinsic_reward": r_int.cpu().numpy().tolist(),
                "observation": obs.tolist(),
                "is_done": dones.tolist(),
            })
        print(f"r_int: {r_int}")
        return obs, r_int.cpu().numpy(), dones, infos
    
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
    # ports = [3000, 3001, 3002, 3003, 3004, 3005, 3006, 3007, 3008, 3009, 3010, 3011, 3012, 3013, 3014]
    ports = [3000]
    env_fns = [make_reach_env_factory(p) for p in ports]
    vec_env = SubprocVecEnv(env_fns)
    wrapped_env = IntrinsicRewardWrapper(vec_env, k=2)

    # directories
    log_dir = "pretrain/logs/dqn"
    model_dir = "pretrain/model/dqn"
    buffer_dir = "pretrain/replay_buffer/dqn"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(buffer_dir, exist_ok=True)

    # Load latest model and buffer if exists
    latest_checkpoint = None
    latest_timestep = 0
    for f in glob.glob(os.path.join(model_dir, "pretrain_reach_sponge_dqn*.zip")):
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
    TOTAL_TIMESTEPS = 1800
    SAVE_INTERVAL = 1800
    for lts in range(latest_timestep, TOTAL_TIMESTEPS, SAVE_INTERVAL):
        model.learn(
            total_timesteps=SAVE_INTERVAL,
            reset_num_timesteps=False,
            tb_log_name="dqn"
        )
        timestep = lts + SAVE_INTERVAL
        checkpoint = os.path.join(model_dir, f"pretrain_reach_sponge_dqn{timestep}.zip")
        buffer = os.path.join(buffer_dir, f"replay_buffer_{timestep}.npz")
        model.save(checkpoint)
        # save_replay_buffer(model.replay_buffer, buf)
        model.replay_buffer.save(buffer, "pretrain")
        print(f"Saved model and buffer at {timestep}")

    wrapped_env.close()