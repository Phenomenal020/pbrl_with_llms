import json
import jsonlines
import numpy as np
import os

from stable_baselines3.common.buffers import ReplayBuffer
from off_policy_reward import RewardModel

reward_model_dir = "enhanced_rl/reward_model/dqn"
os.makedirs(reward_model_dir, exist_ok=True)

class DQNPBRLReplayBuffer(ReplayBuffer):
    
    def __init__(self, *args, reward_model_dir="enhanced_rl/reward_model/dqn", **kwargs):
        super().__init__(*args, **kwargs)
        self.save_step = 232000
        self.load_step = 224000
        self.reward_model = None
        self.last_pos = 14000


    def save(self, path: str, source = "standard"):
        if source != "standard":
            # First, save current replay buffer elems to a jsonl file
            self.save_to_json("enhanced", "before")
            # make sure save path for the RB itself exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            # load the reward model 
            self.reward_model_dir = reward_model_dir
            self.reward_model = RewardModel(ds=7, da=3)
            try:
                loaded_files = self.reward_model.load(model_dir=self.reward_model_dir, load_step=self.load_step)
                if len(loaded_files) > 0:
                    print(f"Loaded reward model from {self.load_step}...")
            except Exception as e:
                print(f"No pre-trained model found at {self.reward_model_dir}: {e}")
            # train the reward model
            is_data = self.reward_model.add_data()
            if is_data:
                # Use a hybrid sampling method 
                self.reward_model.sample_queries(samp_index=np.random.randint(1,7))
                # sample queries also trains the reward model
                # print(f"Saving reward model to {self.pos}...")
                # next, reload the updated reward model with save step
                self.reward_model_dir = reward_model_dir
                self.reward_model = RewardModel(ds=7, da=3)
                try:
                    loaded_files = self.reward_model.load(model_dir=self.reward_model_dir, load_step=self.save_step)
                    if len(loaded_files) > 0:
                        print(f"Loaded reward model from {self.save_step}...")
                except Exception as e:
                    print(f"No pre-trained model found at {self.reward_model_dir}: {e}")
                # next, relabel the rewards in the buffer
                self.relabel_with_predictor()
                # save again to json file - for inspection
                self.save_to_json("enhanced", "after")
                # finally, save new replay buffer 
                np.savez_compressed(path,
                    observations=self.observations,
                    next_observations=self.next_observations,
                    actions=self.actions,
                    rewards=self.rewards,
                    dones=self.dones,
                    timeouts=self.timeouts,
                    pos=self.pos,
                    full=self.full
                )
                print(f"Replay buffer saved to: {path}")
        else:
            np.savez_compressed(path,
                observations=self.observations,
                next_observations=self.next_observations,
                actions=self.actions,
                rewards=self.rewards,
                dones=self.dones,
                timeouts=self.timeouts,
                pos=self.pos,
                full=self.full
            )
            print(f"Replay buffer saved to: {path}")



    def load(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Replay buffer file not found: {path}")
        data = np.load(path, allow_pickle=True)
        self.observations     = data['observations']
        self.next_observations = data['next_observations']
        self.actions          = data['actions']
        self.rewards          = data['rewards']
        self.dones            = data['dones']
        self.timeouts         = data['timeouts']
        self.pos              = int(data['pos'])
        self.full             = bool(data['full'])
        print(f"Replay buffer loaded from: {path}")
        print(f"Buffer capacity (max transitions): {self.buffer_size}")



    def save_to_json(self, source, stage):
        total_entries = self.buffer_size if self.full else self.pos
        start_idx = self.last_pos
        if source == "standard":
            path = "standard_rl/replay_buffers/dqn/replay_buffer.jsonl"
        elif source == "enhanced":
            if stage == "before":
                path = "enhanced_rl/replay_buffer/dqn/replay_buffer_before.jsonl"
            elif stage == "after":
                path = "enhanced_rl/replay_buffer/dqn/replay_buffer_after.jsonl"
            else:
                raise ValueError("Invalid stage provided. Must be 'before' or 'after'.")
        elif source == "pretrain":
            path = "pretrain/replay_buffer/dqn/replay_buffer.jsonl"
        print(f"Saving {total_entries*self.n_envs} transitions to {path}")

        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Write only the recent entries
        with jsonlines.open(path, mode="w") as writer:
            for idx in range(start_idx, total_entries):
                writer.write({
                    "obs":      self.observations[idx].tolist(),
                    "action":   self.actions[idx].tolist(),
                    "reward":   self.rewards[idx].tolist(),
                    "next_obs": self.next_observations[idx].tolist(),
                    "done":     self.dones[idx].tolist(),
                    "timeout":  self.timeouts[idx].tolist()
                })
        print(f"Saved {total_entries - start_idx} new entries to {path}.")
        print(f"Buffer initial size: {self.last_pos} *** (max size): {self.buffer_size}")
        print(f"Buffer current size: {self.pos} *** (max size): {self.buffer_size}")
        
        
        
    def relabel_with_predictor(self, batch_size: int = 500):
        max_idx = self.buffer_size if self.full else self.pos
        if max_idx == 0:
            print("Replay buffer is empty, skipping relabeling.")
            return
        # determine obs_shape dynamically
        obs_shape = self.observations.shape[2:]  # drop (steps, n_envs)
        # Flatten observations (env-major)
        flat_obs = (
            self.observations[:max_idx]              # (steps, n_envs, *obs_shape)
                .swapaxes(0, 1)                      # (n_envs, steps, *obs_shape)
                .reshape(-1, *obs_shape)             # (n_envs*steps, *obs_shape)
        )

        # Flatten existing rewards
        flat_rewards = (
            self.rewards[:max_idx]                   # (steps, n_envs)
                .swapaxes(0, 1)                      # (n_envs, steps)
                .reshape(-1)                         # (n_envs*steps,)
        )

        total_steps = flat_obs.shape[0]
        num_batches = (total_steps + batch_size - 1) // batch_size

        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, total_steps)
            # r_hat_batch returns (batch_size, 1) or (batch_size,)
            new_r_t = self.reward_model.r_hat_batch(flat_obs[start:end])  # Tensor, maybe (batch,1)
            # detach, move to CPU, convert to NumPy, and reshape
            new_r   = new_r_t.detach().cpu().numpy().reshape(-1) 
            flat_rewards[start:end] = new_r

        # write back into the 2D buffer
        self.rewards[:max_idx] = (
            flat_rewards
                .reshape(self.n_envs, max_idx)      # (n_envs, steps)
                .swapaxes(0, 1)                     # (steps, n_envs)
        )