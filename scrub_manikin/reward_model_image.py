# sources: 
# RL-SaLLM-F paper: https://github.com/TU2021/RL-SaLLM-F/blob/main/reward_model.py
# Docstrings are automatically generated using codeium but verified manually for correctness

import numpy as np   # ✅

import torch  # ✅
import torch.nn as nn  # ✅
import torch.nn.functional as F  # ✅
import torch.optim as optim  # ✅

from torchviz import make_dot

import itertools  # ✅
import tqdm  # ✅
import copy  # ✅
import os  # ✅
import time  # ✅
import json  # ✅

import jsonlines

import scipy.stats as st  # ✅
from scipy.stats import norm # ✅

import json

import argparse

from imageapi import (
    traj_pair_process,  stringify_trajs, gpt_infer_no_rag, gpt_infer_rag, get_image_paths
) # ✅

# TODO: Implement discriminator


device = "cuda" if torch.cuda.is_available() else "cpu" # use cuda if available  # ✅

os.makedirs("enhanced_rl/db/responses", exist_ok=True)
os.makedirs("enhanced_rl/db/labels", exist_ok=True)
os.makedirs("enhanced_rl/db/preferences", exist_ok=True)


# -----------------------1. Utility Functions -----------------------

def gen_net(in_size=1, out_size=1, H=128, n_layers=3, activation='relu',
            use_batch_norm=False, use_dropout=True, dropout_prob=0.2):
    net = []
    for i in range(n_layers):
        net.append(nn.Linear(in_size, H))
        if use_batch_norm:
            net.append(nn.BatchNorm1d(H))
        net.append(nn.LeakyReLU())
        if use_dropout:
            net.append(nn.Dropout(p=dropout_prob))
        in_size = H

    net.append(nn.Linear(in_size, out_size))
    return net


class RewardModel:

    def __init__(self, #✅
            ds,                 # state dimension: size of the state/observation vector. ✅
            da,                 # action dimension: size of the action vector. ✅
            ensemble_size=3,    # number of reward predictors in the ensemble ✅
            lr=3e-4,            # learning rate for the optimiser ✅
            # mb_size = 150,      # mini-batch size used during training. ✅#
            mb_size = 10,      # mini-batch size used during ablation. ✅#
            activation='relu',  # activation function to use in the outer layer of the networks ✅
            capacity=1e5,       # total capacity of the preference buffer that stores training transitions ✅
            size_segment=2,    # length of each trajectory segment ✅
            max_size=800,       # maximum number of trajectory segments to keep (for FIFO logic) ✅  # Change to 10 to simulate (10x50) = 500 timesteps.
            large_batch=4,      # multiplier to increase the effective batch size during query sampling ✅
            label_margin=0.0,   # margin used when constructing soft target labels ✅
            teacher_beta=0,    # parameter for a Bradley-Terry model; if > 0, used to compute soft labels. For ablation, set to 0 to use hard rational labels instead. ✅
            teacher_gamma=1,    # decay factor applied to earlier parts of a trajectory when computing “rational” labels ✅
            teacher_eps_mistake=0.0,  # probability of intentionally flipping a label (simulating teacher mistakes) - 5%. For ablation, set to 0.✅
            teacher_eps_skip=0,     # threshold modifier to skip queries with very high rewards ✅
            teacher_eps_equal=0,    # threshold modifier to mark two segments as equally preferable. ✅
            traj_action=False,   # if True, each input will be a concatenation of state and action (trajectory action). ✅
            traj_save_path=None,# file path to save trajectories (for later inspection or debugging). ✅
            vlm_label=None,     # specifies if a visual language model (or similar) is used for labeling. ✅
            better_traj_gen=False,  # flag indicating whether to try to generate an improved trajectory. ✅
            double_check=True,     # if True, perform a double-check (swapped order) of labeling. ✅
            save_equal=True,    # if True, store queries with equal preference labels as given by the teacher. ✅
            vlm_feedback=True,  # flag indicating whether to use feedback from a visual language model. ✅
            generate_check=False,   # flag for additional checks during generation. ✅
            # env_id = None,  # Stores environment ID to track multi-processing 
        ):

        # train data is trajectories, must process to sa and s..   
        
        self.traj_action = traj_action # ✅
        self.traj_save_path = traj_save_path # ✅
        self.ds = ds # ✅
        self.da = da # ✅
        self.de = ensemble_size # ✅
        self.lr = lr # ✅

        self.ensemble = [] # list of reward models in the ensemble ✅
        self.paramlst = [] # list of parameters of each reward model in the ensemble ✅

        self.opt = None # optimiser for the ensemble ✅
        self.model = None  # ✅

        self.max_size = max_size # ✅
        self.activation = activation # ✅
        
        self.size_segment = 5 # ✅
        self.model_index = 2
        self.use_rag = False # ✅

        self.vlm_label = vlm_label # ✅
        self.better_traj_gen = better_traj_gen # ✅
        self.double_check = double_check # ✅
        self.save_equal = save_equal # ✅
        self.vlm_feedback = vlm_feedback # ✅
        self.generate_check = generate_check # ✅
        self.capacity = int(capacity) # ✅

        # self.index_update = True
      
        if traj_action:
            self.obs_buffer_seg1 = np.empty((self.capacity, size_segment, self.ds+self.da), dtype=np.float32) # ✅
            self.obs_buffer_seg2 = np.empty((self.capacity, size_segment, self.ds+self.da), dtype=np.float32) # ✅
        else:
            self.obs_buffer_seg1 = np.empty((self.capacity, size_segment, self.ds), dtype=np.float32) # ✅
            self.obs_buffer_seg2 = np.empty((self.capacity, size_segment, self.ds), dtype=np.float32) # ✅
        self.pred_labels_buffer = np.empty((self.capacity, 1), dtype=np.float32) # ✅
        self.llm_labels_buffer = np.empty((self.capacity, 1), dtype=np.float32) # ✅
        self.true_labels_buffer = np.empty((self.capacity, 1), dtype=np.float32) # ✅

        self.buffer_env_id = np.empty((self.capacity, 1), dtype=np.int32)  # Stores 
        self.fake_flag = np.empty((self.capacity, 1), dtype=bool) # ✅
        self.buffer_index = 0 # ✅
        self.buffer_full = False # ✅

        self.construct_ensemble() # ✅called to create the ensemble of reward predictors. An Adam optimiser is created over the combined parameters of all networks for optimisation.
        
        self.obses = []  # ✅
        self.pred_rewards = [] # ✅
        self.true_rewards = [] # ✅
        self.actions = [] # ✅
        # self.dones = [] # ✅
        self.truncateds = [] # ✅
        # self.collisions = [] # ✅
        # self.successes = [] # ✅
        self.goals = [] # ✅
        self.imageobses = []

        self.mb_size = mb_size # ✅
        self.origin_mb_size = mb_size # ✅
        self.train_batch_size = 64  # ✅

        self.CEloss = nn.CrossEntropyLoss() # ✅

        self.large_batch = large_batch # ✅
        
        # new teacher
        self.teacher_beta = teacher_beta # ✅ 
        self.teacher_gamma = teacher_gamma # ✅
        self.teacher_eps_mistake = teacher_eps_mistake # ✅
        self.teacher_eps_equal = teacher_eps_equal # ✅
        self.teacher_eps_skip = teacher_eps_skip # ✅
        self.teacher_thres_skip = -1 # do not use reward difference to skip queries ✅
        self.teacher_thres_equal = 0 # ✅

        # for llms
        self.llm_query_accuracy = 0 # ✅
        self.llm_label_accuracy = 0 # ✅

        self.device = device # ✅

        print(f"Is temp buffer empty?: {len(self.obses) == 0}")
        
        self.save_step = 4800 # ✅
        self.add_index = 0 # ✅
        
        self.llm_model = "gpt-4o-mini"
        self.threshold_variance='kl'
        self.threshold_alpha=0.5
        self.threshold_beta_init=3.0
        self.threshold_beta_min=1.0
        self.k = 30
        # self.KL_div = RunningMeanStd(mode='fixed', lr=0.1)

    
    def construct_ensemble(self): # ✅
        for i in range(self.de):
            if self.traj_action:
                model = nn.Sequential(*gen_net(in_size=self.ds+self.da, 
                                           out_size=1, H=128, n_layers=3, 
                                           activation=self.activation)).float().to(device)
            else:
                model = nn.Sequential(*gen_net(in_size=self.ds, 
                                           out_size=1, H=128, n_layers=3, 
                                           activation=self.activation)).float().to(device)
            self.ensemble.append(model) 
            self.paramlst.extend(model.parameters())
            
        self.opt = torch.optim.Adam(self.paramlst, lr = self.lr)

    # ------------------------- 3. Trajectory Data Storage --------------------------
# 

    def add_data(self): # ✅
        file_paths = [
            "enhanced_rl/db/trajectories/enhanced_reach_sponge3000.jsonl",
            "enhanced_rl/db/trajectories/enhanced_reach_sponge3001.jsonl",
            "enhanced_rl/db/trajectories/enhanced_reach_sponge3002.jsonl",
            "enhanced_rl/db/trajectories/enhanced_reach_sponge3003.jsonl" ,
            "enhanced_rl/db/trajectories/enhanced_reach_sponge3004.jsonl",
            "enhanced_rl/db/trajectories/enhanced_reach_sponge3005.jsonl",
            "enhanced_rl/db/trajectories/enhanced_reach_sponge3006.jsonl",
            "enhanced_rl/db/trajectories/enhanced_reach_sponge3007.jsonl",
            "enhanced_rl/db/trajectories/enhanced_reach_sponge3008.jsonl",
            "enhanced_rl/db/trajectories/enhanced_reach_sponge3009.jsonl",
            "enhanced_rl/db/trajectories/enhanced_reach_sponge3010.jsonl"
            "enhanced_rl/db/trajectories/enhanced_reach_sponge3011.jsonl",
            "enhanced_rl/db/trajectories/enhanced_reach_sponge3012.jsonl",
            "enhanced_rl/db/trajectories/enhanced_reach_sponge3013.jsonl",
            "enhanced_rl/db/trajectories/enhanced_reach_sponge3014.jsonl",
            "enhanced_rl/db/trajectories/enhanced_reach_sponge3015.jsonl",
            ]  # For ablation
        
        print("Loading trajectory segments from JSONL files...")
        index = 0
        for file_path in file_paths:
            try:
                with open(file_path, "r") as f:
                    # Process each line in the file
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        index += 1
                        
                        # Load the JSON object
                        data = json.loads(line)
                        
                        # If buffer is full, remove the oldest entry from each buffer to maintain fixed size.
                        if len(self.obses) >= self.max_size:
                            self.obses.pop(0)
                            self.actions.pop(0)
                            self.true_rewards.pop(0)
                            self.pred_rewards.pop(0)
                            self.goals.pop(0)
                            self.imageobses.pop(0)
                            # self.dones.pop(0)
                            # self.truncateds.pop(0)
                            # self.collisions.pop(0)
                            # self.successes.pop(0)
                        
                        # Append the new trajectory data.
                        # self.obses.append(data["obs"])
                        self.obses.append(data["obs"][-self.size_segment:])
                        self.actions.append(data["act"][-self.size_segment:])
                        self.true_rewards.append(data["true_rew"][-self.size_segment:])
                        self.pred_rewards.append(data["pred_rew"][-self.size_segment:])
                        self.imageobses.append(data["imageobs"][-self.size_segment:])
                        # self.dones.append(data["done"])
                        self.truncateds.append(data["truncated"][-self.size_segment:])
                        self.goals.append(data["goal"][-self.size_segment:])
                        # self.collisions.append(data["collision"])
                        # self.successes.append(data["success"])
                        
                    # Log that we finished processing a file.
                    # print(f"Finished loading file: {file_path}")
            
            except FileNotFoundError:
                error_msg = f"File not found: {file_path}"
                # print(error_msg)
            except json.JSONDecodeError as e:
                error_msg = f"Error decoding JSON in file {file_path}: {e}"
                # print(error_msg)
            except Exception as e:
                error_msg = f"An error occurred while reading {file_path}: {e}"
                # print(error_msg)

        print(f"Temporary buffer size: {len(self.obses)}, {index}")
        self.add_index = min(len(self.obses), self.capacity)
        print(f"Add data index: {self.add_index}")
        # self.buffer_full = self.buffer_index >= self.capacity
        return True

    # --------------------------4. Probability and rewards --------------------------

    # Computes reward for a given input and ensemble member.
    # Called by r_hat and r_hat_batch self.de times
    def r_hat_member(self, x, member=-1): # ✅
        # Check if x is already a tensor, if not convert it.
        # This is necessary because the ensemble models expect a tensor as input.
        if not isinstance(x, torch.Tensor):
            # Check if x is a scalar (int or float)
            if np.isscalar(x):
                x = torch.tensor([x])
            # Otherwise, assume it is a NumPy array
            else:
                x = torch.from_numpy(x)
                
        x = x.float().to(self.device)
        return self.ensemble[member](x.unsqueeze(0))
    
    def r_hat_member_no_unsqueeze(self, x, member=-1): # ✅
        # Check if x is already a tensor, if not convert it.
        # This is necessary because the ensemble models expect a tensor as input.
        if not isinstance(x, torch.Tensor):
            # Check if x is a scalar (int or float)
            if np.isscalar(x):
                x = torch.tensor([x])
            # Otherwise, assume it is a NumPy array
            else:
                x = torch.from_numpy(x)
                
        x = x.float().to(self.device)
        return self.ensemble[member](x)


    # Computes the average reward for a given input and all ensemble members
    def r_hat(self, x): # ✅
        r_hats = []
        for member in range(self.de):
            r_hats.append(self.r_hat_member(x, member=member))
        r_hats = torch.stack(r_hats, dim=0)
        return r_hats.mean(dim=0) / 10

    
    # Computes the average reward for a given batch of inputs and returns a vector of rewards corresponding to the batched mean reward for all members
    def r_hat_batch(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x)
        x = x.float().to(self.device)

        r_hats = []
        for member in range(self.de):
            member_reward = self.r_hat_member(x, member=member)  # should return a tensor with grad
            r_hats.append(member_reward)
        # Stack and mean the results across the ensemble dimension
        r_hats = torch.stack(r_hats, dim=0)  # shape: (de, batch_size)
        return torch.mean(r_hats, dim=0)     # shape: (batch_size,)


    def r_hat_batch_single_member(self, x, member=0):
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x)

        x = x.float().to(self.device)
        batch_size = x.shape[0]
        rewards = []

        for b in range(batch_size):
            segment = x[b]  # (time_steps, features)

            r_hat = self.r_hat_member(segment, member=member)            
            r_hat_mean = r_hat.mean()
            rewards.append(r_hat_mean)

        rewards = torch.stack(rewards)
        return rewards


    def get_rank_probability(self, x_1, x_2):
        # get probability x_1 > x_2
        probs = []
        for member in range(self.de):
            probs.append(self.p_hat_member(x_1, x_2, member=member).cpu().numpy())
        probs = np.array(probs)
        return np.mean(probs, axis=0), np.std(probs, axis=0)
    
    def p_hat_member(self, x_1, x_2, member=-1):
        # Probability that x_1 is preferred over x_2
        with torch.no_grad():
            r_hat1 = self.r_hat_batch_single_member(x_1, member=member)
            r_hat2 = self.r_hat_batch_single_member(x_2, member=member)
            # print(f"rhat1: {r_hat1}")
            # print(f"rhat2: {r_hat2}")
            # print(r_hat1.shape, r_hat2.shape)
            r_hat = torch.cat([r_hat1.unsqueeze(1), r_hat2.unsqueeze(1)], dim=1)
        return F.softmax(r_hat, dim=1)[:, 0] # use softmax to compute probabilities

    def p_hat_entropy(self, x_1, x_2, member=-1):
        # Entropy of the preference distribution
        with torch.no_grad():
            r_hat1 = self.r_hat_member(x_1, member=member).sum(dim=1)
            r_hat2 = self.r_hat_member(x_2, member=member).sum(dim=1)
            r_hat = torch.cat([r_hat1.unsqueeze(1), r_hat2.unsqueeze(1)], dim=1)
            probs = F.softmax(r_hat, dim=1)
            log_probs = F.log_softmax(r_hat, dim=1)
            ent = -torch.sum(probs * log_probs, dim=1)  # standard entropy
        return ent



    # --------------------------5. Checkpointing --------------------------

    def save(self, model_dir="enhanced_rl/reward_model/dqn", save_step=None): # ✅
        print("Saving reward model...")

        # Ensure directory exists. If not, create it.
        os.makedirs(model_dir, exist_ok=True)

        saved_files = []
        if save_step is None:
            save_step = self.save_step
        for member in range(self.de):
            file_path = os.path.join(model_dir, f"reward_model_{save_step}_{member}.pt")
            torch.save(self.ensemble[member].state_dict(), file_path)
            saved_files.append(file_path)

        print("Reward model saved successfully.")

        return saved_files  # Return list of saved file paths
    
    

    def load(self, model_dir="enhanced_rl/reward_model/dqn", load_step=0): # ✅
        print("Loading reward model...")

        loaded_files = []
        for member in range(self.de):
            file_path = os.path.join(model_dir, f"reward_model_{load_step}_{member}.pt")

            if not os.path.exists(file_path):
                print(f"Warning: Reward model file {file_path} not found.")
                continue  # Skip missing models instead of crashing

            with torch.no_grad():  # Prevent tracking gradients while loading
                self.ensemble[member].load_state_dict(torch.load(file_path))

            loaded_files.append(file_path)
        
        # print(f"loaded_files: {loaded_files}") # (loaded_files)
        if len(loaded_files) > 0:
            print(f"Reward model with load_step {load_step} loaded successfully.")

        return loaded_files  # Return list of loaded file paths
    


    # --------------------------6. Getting Labels --------------------------

    def sample_indices(self, sample_buffer, mb_size,  rng=None): # ✅
        if rng is None:
            rng = np.random.default_rng()

        buffer = np.array(sample_buffer)

        # Verify that the buffer has at least one segment with the required length.
        if len(buffer) < 1 or (len(buffer[0]) < self.size_segment):
            raise ValueError(f"Sample buffer does not contain any segment of the required length {self.size_segment}")

        indices = rng.choice(len(buffer), size=mb_size, replace=True)

        return indices


    def get_queries_uniform(self, mb_size=None): # ✅
        print("Getting queries...")

        # Instantiate a dedicated random generator with the provided seed.
        rng = np.random.default_rng()

        # Check for available trajectory data.
        if len(self.obses) == 0:
            raise ValueError("No trajectory data available.")

        # Sample indices for the first set using the provided rng.
        if mb_size != None:
            indices1 = self.sample_indices(self.obses, mb_size, rng=rng)
             # Compute the set of remaining indices - not chosen in indices1.
            all_indices = np.arange(len(self.obses))
            remaining_indices = np.setdiff1d(all_indices, indices1)

            # Sample new indices for the second set from the remaining indices.
            if len(remaining_indices) < mb_size:
                # Not enough unique segments are available; sample with replacement.
                indices2 = rng.choice(remaining_indices, size=mb_size, replace=True)
            else:
                indices2 = rng.choice(remaining_indices, size=mb_size, replace=True)
        else:
            indices1 = self.sample_indices(self.obses, self.mb_size, rng=rng)
            # Compute the set of remaining indices - not chosen in indices1.
            all_indices = np.arange(len(self.obses))
            remaining_indices = np.setdiff1d(all_indices, indices1)

            # Sample new indices for the second set from the remaining indices.
            if len(remaining_indices) < self.mb_size:
                # Not enough unique segments are available; sample with replacement.
                indices2 = rng.choice(remaining_indices, size=self.mb_size, replace=True)
            else:
                indices2 = rng.choice(remaining_indices, size=self.mb_size, replace=True)
        obs1 = np.array(self.obses)[indices1]
        acts1 = np.array(self.actions)[indices1]
        true_r1 = np.array(self.true_rewards)[indices1]
        pred_r1 = np.array(self.pred_rewards)[indices1]
        # dones1 = np.array(self.dones)[indices1]
        truncated1 = np.array(self.truncateds)[indices1]
        # collisions1 = np.array(self.collisions)[indices1]
        # successes1 = np.array(self.successes)[indices1]
        goals1 = np.array(self.goals)[indices1]
        imageobses1 = np.array(self.imageobses)[indices1]

        obs2 = np.array(self.obses)[indices2]
        acts2 = np.array(self.actions)[indices2]
        true_r2 = np.array(self.true_rewards)[indices2]
        pred_r2 = np.array(self.pred_rewards)[indices2]
        # dones2 = np.array(self.dones)[indices2]
        truncated2 = np.array(self.truncateds)[indices2]
        # collisions2 = np.array(self.collisions)[indices2]
        # successes2 = np.array(self.successes)[indices2]
        goals2 = np.array(self.goals)[indices2]
        imageobses2 = np.array(self.imageobses)[indices2]

        print("Got queries successfully using uniform sampling..")
        
        print(f"obs1: {obs1.shape}")
        print(f"obs2: {obs2.shape}")

        return (obs1, obs2, 
                acts1, acts2, 
                true_r1, true_r2, 
                pred_r1, pred_r2, 
                # dones1, dones2, 
                truncated1, truncated2, 
                # collisions1, collisions2, 
                # successes1, successes2
                goals1, goals2,
                imageobses1, imageobses2
                )
        
    
        
        
    
    def get_entropy(self, x1, x2):
        # Collect logits from each ensemble member: shape (de, batch_size, 2)
        all_logits = []
        with torch.no_grad():
            for m in range(self.de):
                # Model outputs a score/logit vector; sum across feature dims
                r1 = self.r_hat_batch_single_member(x1, member=m)
                r2 = self.r_hat_batch_single_member(x2, member=m)
                # Stack scores so dim=-1 has [score_x1, score_x2]
                all_logits.append(torch.stack([r1, r2], dim=-1))
        all_logits = torch.stack(all_logits, dim=0)  # (de, batch, 2)

        # Softmax over the last axis to get P(x1 > x2) per member
        p = F.softmax(all_logits, dim=-1)[:, :, 0]  # (de, batch)
        p = p.cpu().numpy()                         # convert to NumPy

        # Mean probability across members, then Shannon entropy
        p_mean = p.mean(axis=0)                     # (batch,)
        entropy = -(p_mean * np.log(p_mean + 1e-12)
                    + (1 - p_mean) * np.log(1 - p_mean + 1e-12))

        # Standard deviation of probabilities as auxiliary uncertainty
        std_dev = p.std(axis=0)                     # (batch,)
        return entropy, std_dev
    

    def get_queries_entropy(self, seed=42):
        # Determine pool size and clamp to available data
        pool_size = self.mb_size * self.large_batch
        total = len(self.obses)
        print(f"Pool size: {pool_size}, total: {total}")
        pool_size = min(pool_size, total)

        # Sample first view indices
        rng = np.random.default_rng(seed)
        idx1 = self.sample_indices(self.obses, pool_size, rng=rng)

        # Sample second view indices from remaining
        all_idx = np.arange(total)
        remaining = np.setdiff1d(all_idx, idx1)
        if remaining.size == 0:
            # All were selected once: sample from full set
            idx2 = rng.choice(all_idx, size=pool_size, replace=True)
        elif remaining.size < pool_size:
            # Not enough unique samples: sample with replacement from remaining
            idx2 = rng.choice(remaining, size=pool_size, replace=True)
        else:
            # Enough unique samples: sample without replacement
            idx2 = rng.choice(remaining, size=pool_size, replace=False)
            
        print(f"idx1: {idx1.shape}, idx2: {idx2.shape}")

        # Convert observations to torch tensors for entropy computation
        x1 = torch.tensor(np.array(self.obses)[idx1],
                          device=self.device, dtype=torch.float32)
        x2 = torch.tensor(np.array(self.obses)[idx2],
                          device=self.device, dtype=torch.float32)
        
        print(f"x1: {x1.shape}, x2: {x2.shape}")

        # Compute entropy of each candidate pair
        entropy_vals, _ = self.get_entropy(x1, x2)
        entropy_vals = np.squeeze(entropy_vals) 

        # Select top-self.mb_size most uncertain pairs
        top_k = np.argsort(-entropy_vals)[: self.mb_size]
        print("top_k:", top_k, "shape:", top_k.shape)


        # Gather and return only those top-k pairs across all buffers
        def _slice(buffer, idx):
            buf_arr = np.array(buffer)
            print(f"buf_arr: {buf_arr.shape}")
            return buf_arr[idx][top_k]

        # Apply the _slice function using both idx1 and idx2
        obs1, obs2 = _slice(self.obses, idx1), _slice(self.obses, idx2)
        acts1, acts2 = _slice(self.actions, idx1), _slice(self.actions, idx2)
        true_r1, true_r2 = _slice(self.true_rewards, idx1), _slice(self.true_rewards, idx2)
        pred_r1, pred_r2 = _slice(self.pred_rewards, idx1), _slice(self.pred_rewards, idx2)
        # dones1, dones2 = _slice(self.dones, idx1), _slice(self.dones, idx2)
        trunc1, trunc2 = _slice(self.truncateds, idx1), _slice(self.truncateds, idx2)
        # coll1, coll2 = _slice(self.collisions, idx1), _slice(self.collisions, idx2)
        # succ1, succ2 = _slice(self.successes, idx1), _slice(self.successes, idx2)   
        goals1, goals2 = _slice(self.goals, idx1), _slice(self.goals, idx2)  
        imageobses1, imageobses2 = _slice(self.imageobses, idx1), _slice(self.imageobses, idx2)  
        
        print("Got queries successfully via entropy sampling.")
        print(f"obs1: {obs1.shape}")
        print(f"obs2: {obs2.shape}")
        

        return (
            obs1, obs2,
            acts1, acts2,
            true_r1, true_r2,
            pred_r1, pred_r2,
            # dones1, dones2,
            trunc1, trunc2,
            # coll1, coll2,
            # succ1, succ2,
            goals1, goals2,
            imageobses1, imageobses2
        )


    # -------------------------------------- KCenter Sampling --------------------------------------

    
    def KCenterGreedy(self, obs, full_obs, num_new_sample):
        selected_index = []
        current_index = list(range(obs.shape[0]))
        new_obs = obs
        print(f"kcenter greedy new obs: {new_obs}")
        new_full_obs = full_obs
        print(f"kcenter greedy new full obs: {new_full_obs}")

        for _ in range(num_new_sample):
            dist = self.compute_smallest_dist(new_obs, new_full_obs)
            print(f"kcenter greedy dist: {dist}")
            max_index = torch.argmax(dist).item()
            print(f"kcenter greedy max index: {max_index}")

            # Always index into current_index to get the global index
            chosen = current_index[max_index]
            selected_index.append(chosen)
            print(f"kcenter greedy selected index: {selected_index}")

            # Remove selected index
            current_index.pop(max_index)

            # Update candidate pool
            new_obs = obs[current_index]
            new_full_obs = np.concatenate([
                full_obs,
                obs[selected_index]
            ], axis=0)

        return selected_index


    def compute_smallest_dist(self, obs, full_obs, device='cuda' if torch.cuda.is_available() else 'cpu'):
        obs = torch.from_numpy(obs).float().to(device)
        full_obs = torch.from_numpy(full_obs).float().to(device)
        batch_size = 100
        total_dists = []

        with torch.no_grad():
            for full_idx in range(0, len(obs), batch_size):
                obs_batch = obs[full_idx:full_idx + batch_size]
                dists = []
                for idx in range(0, len(full_obs), batch_size):
                    full_batch = full_obs[idx:idx + batch_size]
                    dist = torch.norm(
                        obs_batch[:, None, :] - full_batch[None, :, :],
                        dim=-1, p=2
                    )
                    dists.append(dist)
                dists = torch.cat(dists, dim=1)  # concat along full_obs dimension
                small_dists = torch.min(dists, dim=1).values  # min distance for each obs
                total_dists.append(small_dists)

        return torch.cat(total_dists)  # shape (N,)

    
    def get_queries_kcenter(self, seed=42):
        pool_size = min(self.mb_size * self.large_batch, len(self.obses))
        rng = np.random.default_rng(seed)
        idx1 = self.sample_indices(self.obses, pool_size, rng=rng)
        idx2 = self.sample_indices(self.obses, pool_size, rng=rng)

        x1 = np.array(self.obses)[idx1].reshape(pool_size, -1)
        x2 = np.array(self.obses)[idx2].reshape(pool_size, -1)
        obs_pairs = np.concatenate([x1, x2], axis=1)

        max_len = self.capacity if self.buffer_full else self.add_index
        if max_len > 0:
            buf1 = np.array(self.obs_buffer_seg1[:max_len]).reshape(max_len, -1)
            buf2 = np.array(self.obs_buffer_seg2[:max_len]).reshape(max_len, -1)
            buffer_obs_pairs = np.concatenate([buf1, buf2], axis=1)
        else:
            # Force K‑Center to run on the pool itself
            buffer_obs_pairs = obs_pairs.copy()
            print("K-Center running on the pool itself.")

        selected_idx = self.KCenterGreedy(obs_pairs, buffer_obs_pairs, self.mb_size)
        # ensure we get exactly mb_size
        if len(selected_idx) < self.mb_size:
            avail = np.setdiff1d(np.arange(pool_size), selected_idx)
            pad  = rng.choice(avail, size=self.mb_size-len(selected_idx), replace=False)
            selected_idx = np.concatenate([selected_idx, pad])

        # now slice everything in one go
        sel1 = idx1[selected_idx]
        sel2 = idx2[selected_idx]
        def gather(arr): 
            a = np.array(arr)
            return a[sel1], a[sel2]

        obs1, obs2     = gather(self.obses)
        acts1, acts2   = gather(self.actions)
        true_r1, true_r2 = gather(self.true_rewards)
        pred_r1, pred_r2 = gather(self.pred_rewards)
        # dones1, dones2   = gather(self.dones)
        trunc1, trunc2   = gather(self.truncateds)
        # coll1, coll2     = gather(self.collisions)
        # succ1, succ2     = gather(self.successes)
        goals1, goals2   = gather(self.goals)
        imageobses1, imageobses2 = gather(self.imageobses)

        return (obs1, obs2, 
                acts1, acts2,
                true_r1, true_r2,
                pred_r1, pred_r2,
                # dones1, dones2, 
                trunc1, trunc2,
                # coll1, coll2, 
                # succ1, succ2
                goals1, goals2,
                imageobses1, imageobses2
                )


    def kcenter_disagreement_sampling(self):
        num_init = self.mb_size * self.large_batch
        num_init_half = int(num_init * 0.5)

        # Step 1: Uniformly sample (get full obs + actions + rewards)
        obs1, obs2, acts1, acts2, true_r1, true_r2, pred_r1, pred_r2, trunc1, trunc2, goals1, goals2, imageobses1, imageobses2 = self.get_queries_uniform(mb_size=num_init)

        # Step 2: Rank by disagreement over reward trajectories
        _, disagreement = self.get_rank_probability(obs1, obs2)  # disagreement = standard deviation
        # print(f"disagreement: {disagreement}, {disagreement.shape}")
        top_k_index = (-disagreement).argsort()[:num_init_half]
        
        # print(f"top k index: {top_k_index}, {top_k_index.shape}")

        # Step 3: Filter top-k
        obs1 = obs1[top_k_index]
        obs2 = obs2[top_k_index]
        acts1 = acts1[top_k_index]
        acts2 = acts2[top_k_index]
        true_r1 = true_r1[top_k_index]
        true_r2 = true_r2[top_k_index]
        pred_r1 = pred_r1[top_k_index]
        pred_r2 = pred_r2[top_k_index]
        # dones1 = dones1[top_k_index]
        # dones2 = dones2[top_k_index]
        trunc1 = trunc1[top_k_index]
        trunc2 = trunc2[top_k_index]
        # coll1 = coll1[top_k_index]
        # coll2 = coll2[top_k_index]
        # succ1 = succ1[top_k_index]
        # succ2 = succ2[top_k_index]
        goals1 = goals1[top_k_index]
        goals2 = goals2[top_k_index]
        imageobses1 = imageobses1[top_k_index]
        imageobses2 = imageobses2[top_k_index]

        # Step 4: Form feature vectors for K-Center Greedy
        batch, T, D = obs1.shape
        # flatten each segment to (batch, T*D)
        flat1 = obs1.reshape(batch, T * D)
        flat2 = obs2.reshape(batch, T * D)
        # concatenate side by side → (batch, 2*T*D)
        temp_r = np.concatenate([flat1, flat2], axis=1)
        # print(f"temp_r: {temp_r.shape}")  # should print (300, 120)

        max_len = self.capacity if self.buffer_full else self.add_index
        tot_r_1 = np.array(self.obses[:max_len]).reshape(max_len, T * D)
        tot_r_2 = np.array(self.obses[:max_len]).reshape(max_len, T * D)
        # Finally concatenate → (max_len, 2*T*D)
        tot_r = np.concatenate([tot_r_1, tot_r_2], axis=1)
        # print(f"tot_r: {tot_r.shape}")

        # Step 5: K-Center selection
        selected_indices = self.KCenterGreedy(temp_r, tot_r, self.mb_size)
        # print(f"Selected indices: {selected_indices}")

        # Step 6: Apply the selected indices to filter observations
        obs1 = obs1[selected_indices]
        obs2 = obs2[selected_indices]
        acts1 = acts1[selected_indices]
        acts2 = acts2[selected_indices]
        true_r1 = true_r1[selected_indices]
        true_r2 = true_r2[selected_indices]
        pred_r1 = pred_r1[selected_indices]
        pred_r2 = pred_r2[selected_indices]
        # dones1 = dones1[selected_indices]
        # dones2 = dones2[selected_indices]
        trunc1 = trunc1[selected_indices]
        trunc2 = trunc2[selected_indices]
        # coll1 = coll1[selected_indices]
        # coll2 = coll2[selected_indices]
        # succ1 = succ1[selected_indices]
        # succ2 = succ2[selected_indices]
        goals1 = goals1[selected_indices]
        goals2 = goals2[selected_indices]
        imageobses1 = imageobses1[selected_indices]
        imageobses2 = imageobses2[selected_indices]
        
        print(f"Queries gotten by kcenter disagreement sampling --> {obs1.shape}")


        return obs1, obs2, acts1, acts2, true_r1, true_r2, pred_r1, pred_r2, trunc1, trunc2, goals1, goals2, imageobses1, imageobses2


    def kcenter_entropy_sampling(self):
        num_init = self.mb_size * self.large_batch
        num_init_half = int(num_init * 0.5)

        # Step 1: Generate candidate queries (get full obs, actions, rewards)
        obs1, obs2, acts1, acts2, true_r1, true_r2, pred_r1, pred_r2, trunc1, trunc2, goals1, goals2, imageobses1, imageobses2 = self.get_queries_uniform(mb_size=num_init)

        # Step 2: Rank by entropy (model uncertainty)
        entropy, _ = self.get_entropy(obs1, obs2)
        top_k_index = (-entropy).argsort()[:num_init_half]

        # Step 3: Filter most uncertain samples
        obs1 = obs1[top_k_index]
        obs2 = obs2[top_k_index]
        acts1 = acts1[top_k_index]
        acts2 = acts2[top_k_index]
        true_r1 = true_r1[top_k_index]
        true_r2 = true_r2[top_k_index]
        pred_r1 = pred_r1[top_k_index]
        pred_r2 = pred_r2[top_k_index]
        # dones1 = dones1[top_k_index]
        # dones2 = dones2[top_k_index]
        trunc1 = trunc1[top_k_index]
        trunc2 = trunc2[top_k_index]
        # coll1 = coll1[top_k_index]
        # coll2 = coll2[top_k_index]
        # succ1 = succ1[top_k_index]
        # succ2 = succ2[top_k_index]
        goals1 = goals1[top_k_index]
        goals2 = goals2[top_k_index]
        imageobses1 = imageobses1[top_k_index]
        imageobses2 = imageobses2[top_k_index]

        # Step 4: Form feature vectors for K-Center Greedy
        batch, T, D = obs1.shape
        # flatten each segment to (batch, T*D)
        flat1 = obs1.reshape(batch, T * D)
        flat2 = obs2.reshape(batch, T * D)
        # concatenate side by side → (batch, 2*T*D)
        temp_r = np.concatenate([flat1, flat2], axis=1)
        # print(f"temp_r: {temp_r.shape}")  # should print (300, 120)

        max_len = self.capacity if self.buffer_full else self.add_index
        tot_r_1 = np.array(self.obses[:max_len]).reshape(max_len, T * D)
        tot_r_2 = np.array(self.obses[:max_len]).reshape(max_len, T * D)
        # Finally concatenate → (max_len, 2*T*D)
        tot_r = np.concatenate([tot_r_1, tot_r_2], axis=1)
        # print(f"tot_r: {tot_r.shape}")
        # Step 5: K-Center selection
        selected_indices = self.KCenterGreedy(temp_r, tot_r, self.mb_size)

        # Step 6: Apply the selected indices to filter observations
        obs1 = obs1[selected_indices]
        obs2 = obs2[selected_indices]
        acts1 = acts1[selected_indices]
        acts2 = acts2[selected_indices]
        true_r1 = true_r1[selected_indices]
        true_r2 = true_r2[selected_indices]
        pred_r1 = pred_r1[selected_indices]
        pred_r2 = pred_r2[selected_indices]
        # dones1 = dones1[selected_indices]
        # dones2 = dones2[selected_indices]
        trunc1 = trunc1[selected_indices]
        trunc2 = trunc2[selected_indices]
        # coll1 = coll1[selected_indices]
        # coll2 = coll2[selected_indices]
        # succ1 = succ1[selected_indices]
        # succ2 = succ2[selected_indices]
        goals1 = goals1[selected_indices]
        goals2 = goals2[selected_indices]
        imageobses1 = imageobses1[selected_indices]
        imageobses2 = imageobses2[selected_indices]

        # # Step 7: Label and store
        # obs1, obs2, acts1, acts2, true_r1, true_r2, pred_r1, pred_r2, dones1, dones2, trunc1, trunc2, coll1, coll2, succ1, succ2, pred_labels, llm_labels = self.get_label(
        #     obs1, obs2, acts1, acts2, true_r1, true_r2, pred_r1, pred_r2, dones1, dones2, trunc1, trunc2, coll1, coll2, succ1, succ2)
        print(f"Queries gotten by kcenter entropy sampling --> {obs1.shape}")

        return obs1, obs2, acts1, acts2, true_r1, true_r2, pred_r1, pred_r2,  trunc1, trunc2, goals1, goals2, imageobses1, imageobses2
    
    def disagreement_sampling(self):
        # Step 1: Generate candidate queries (full obs + actions + rewards)
        obs1, obs2, acts1, acts2, true_r1, true_r2, pred_r1, pred_r2,  trunc1, trunc2, goals1, goals2, imageobses1, imageobses2 = self.get_queries_uniform()
        mb_size = self.mb_size * self.large_batch

        # Step 2: Rank by disagreement score (model uncertainty)
        _, disagreement = self.get_rank_probability(obs1 , obs2)
        print(f"disagreement: {disagreement}, {disagreement.shape}")
        top_k_index = (-disagreement).argsort()[:mb_size]
        print(f"top_k_index: {top_k_index}, {top_k_index.shape}")

        # Step 3: Filter most uncertain samples
        obs1 = obs1[top_k_index]
        obs2 = obs2[top_k_index]
        acts1 = acts1[top_k_index]
        acts2 = acts2[top_k_index]
        true_r1 = true_r1[top_k_index]
        true_r2 = true_r2[top_k_index]
        pred_r1 = pred_r1[top_k_index]
        pred_r2 = pred_r2[top_k_index]
        # dones1 = dones1[top_k_index]
        # dones2 = dones2[top_k_index]
        trunc1 = trunc1[top_k_index]
        trunc2 = trunc2[top_k_index]
        # coll1 = coll1[top_k_index]
        # coll2 = coll2[top_k_index]
        # succ1 = succ1[top_k_index]
        # succ2 = succ2[top_k_index]
        goals1 = goals1[top_k_index]
        goals2 = goals2[top_k_index]
        imageobses1 = imageobses1[top_k_index]
        imageobses2 = imageobses2[top_k_index]
        
        print(f"Queries gotten by disagreement sampling --> {obs1.shape}")

        return obs1, obs2, acts1, acts2, true_r1, true_r2, pred_r1, pred_r2,  trunc1, trunc2, goals1, goals2, imageobses1, imageobses2


    # This method stores the query pairs (and their labels) into circular buffers. It handles buffer overflow with a FIFO (First-In-First-Out) mechanism.
    def put_queries(self):  # ✅
        file_path = "enhanced_rl/db/preferences/preferences.jsonl"
        try:
            with open(file_path, "r") as f:
                preferences = [json.loads(line) for line in f]
            print(f"Successfully read {len(preferences)} preferences.")
        except FileNotFoundError:
            print("Preferences file not found.")
            return 0

        # Aggregate data across all preference entries
        all_obs1 = [p["observations1"] for p in preferences]
        all_obs2 = [p["observations2"] for p in preferences]
        all_pred_labels = [p["prediction_label"] for p in preferences]
        all_llm_labels = [p["llm_label"] for p in preferences]
        all_true_labels = [p["true_label"] for p in preferences]

        # Convert to numpy arrays
        all_obs1 = np.array(all_obs1)
        all_obs2 = np.array(all_obs2)
        all_pred_labels = np.array(all_pred_labels).reshape(-1, 1)
        all_llm_labels = np.array(all_llm_labels).reshape(-1, 1)
        all_true_labels = np.array(all_true_labels).reshape(-1, 1)

        num_samples = all_obs1.shape[0]
        start = self.buffer_index
        print(f"Start index: {start}")
        capacity = self.capacity
        end = start + num_samples
        print(f"End index: {end}")

        # print(f"Capacity of obs buffer: {capacity}, current index: {start}, incoming samples: {num_samples}")
        
        # print(f"Obs1 shape: {all_obs1.shape}, obs2 shape: {all_obs2.shape}, pred_labels shape: {all_pred_labels.shape}, llm_labels shape: {all_llm_labels.shape}, true_labels shape: {all_true_labels.shape}")
        # print("*****************************************")

        # Wrap-around logic
        if end > capacity:
            # Number of slots until end of buffer
            first_chunk = capacity - start
            second_chunk = num_samples - first_chunk

            # Fill first chunk
            np.copyto(self.obs_buffer_seg1[start:capacity], all_obs1[:first_chunk])
            np.copyto(self.obs_buffer_seg2[start:capacity], all_obs2[:first_chunk])
            np.copyto(self.pred_labels_buffer[start:capacity], all_pred_labels[:first_chunk])
            np.copyto(self.llm_labels_buffer[start:capacity], all_llm_labels[:first_chunk])
            np.copyto(self.true_labels_buffer[start:capacity], all_true_labels[:first_chunk])

            # Fill wrapped-around chunk
            if second_chunk > 0:
                np.copyto(self.obs_buffer_seg1[0:second_chunk], all_obs1[first_chunk:])
                np.copyto(self.obs_buffer_seg2[0:second_chunk], all_obs2[first_chunk:])
                np.copyto(self.pred_labels_buffer[0:second_chunk], all_pred_labels[first_chunk:])
                np.copyto(self.llm_labels_buffer[0:second_chunk], all_llm_labels[first_chunk:])
                np.copyto(self.true_labels_buffer[0:second_chunk], all_true_labels[first_chunk:])

            self.buffer_full = True
            self.buffer_index = second_chunk
        else:
            # No wrap needed
            # print("no wrap needed ran")
            np.copyto(self.obs_buffer_seg1[start:end], all_obs1)
            np.copyto(self.obs_buffer_seg2[start:end], all_obs2)
            np.copyto(self.pred_labels_buffer[start:end], all_pred_labels)
            np.copyto(self.llm_labels_buffer[start:end], all_llm_labels)
            np.copyto(self.true_labels_buffer[start:end], all_true_labels)
            

            self.buffer_index = end
            
        # print(f"obs buffer shape: {self.obs_buffer_seg1.shape}")
        # print(f"pred labels buffer shape: {self.pred_labels_buffer.shape}")
        # print(f"llm labels buffer shape: {self.llm_labels_buffer.shape}")
        # print(f"true labels buffer shape: {self.true_labels_buffer.shape}")
        # print("******************************************")

        print("Stored queries successfully...")
        print(f"Buffer index: {self.buffer_index} / {capacity} (full={self.buffer_full})")

        return num_samples



    # assign labels (or preferences) to a pair of trajectory segments based on their predicted rewards   
    def get_label(self, obs1, obs2, acts1, acts2, true_r1, true_r2, pred_r1, pred_r2,truncated1, truncated2, goals1, goals2, imageobses1, imageobses2): # ✅
       # Squeeze out the extra dimensions for pred_r1 and pred_r2
        pred_r1 = np.squeeze(pred_r1)
        pred_r2 = np.squeeze(pred_r2)
        true_r1 = np.squeeze(true_r1)
        true_r2 = np.squeeze(true_r2)
    
        # Compute the segment reward as the difference between the final and initial reward.
        segment_reward_true1 = true_r1[:, -1] - true_r1[:, 0]
        segment_reward_true2 = true_r2[:, -1] - true_r2[:, 0]
        segment_reward_pred1 = pred_r1[:, -1] - pred_r1[:, 0]
        segment_reward_pred2 = pred_r2[:, -1] - pred_r2[:, 0]

        # Filter out queries where both segments are not informative enough.
        if self.teacher_thres_skip > 0:
            max_true_r = np.maximum(segment_reward_true1, segment_reward_true2)
            max_pred_r = np.maximum(segment_reward_pred1, segment_reward_pred2)
            valid_true_mask = max_true_r > self.teacher_thres_skip
            valid_pred_mask = max_pred_r > self.teacher_thres_skip
            if np.sum(valid_true_mask) == 0 and np.sum(valid_pred_mask) == 0:
                return None, None, None, None, None, None, [], [], []
        else:
            # If teacher_thres_skip is not positive (ie, not being used), all queries are valid.
            valid_true_mask = np.ones_like(segment_reward_true1, dtype=bool)
            valid_pred_mask = np.ones_like(segment_reward_pred1, dtype=bool)

        # Filter arrays for true rewards and predicted rewards separately.
        obs1_true = obs1[valid_true_mask]
        obs2_true = obs2[valid_true_mask]
        acts1_true = acts1[valid_true_mask]
        acts2_true = acts2[valid_true_mask]
        true_r1_filt = true_r1[valid_true_mask]
        true_r2_filt = true_r2[valid_true_mask]
        # dones1_true = dones1[valid_true_mask]
        # dones2_true = dones2[valid_true_mask]
        truncated1_true = truncated1[valid_true_mask]
        truncated2_true = truncated2[valid_true_mask]
        # collisions1_true = collisions1[valid_true_mask]
        # collisions2_true = collisions2[valid_true_mask]
        # successes1_true = successes1[valid_true_mask]
        # successes2_true = successes2[valid_true_mask]
        goals1_true = goals1[valid_true_mask]
        goals2_true = goals2[valid_true_mask]
        imageobses1_true = imageobses1[valid_true_mask]
        imageobses2_true = imageobses2[valid_true_mask]

        obs1_pred = obs1[valid_pred_mask]
        obs2_pred = obs2[valid_pred_mask]
        acts1_pred = acts1[valid_pred_mask]
        acts2_pred = acts2[valid_pred_mask]
        pred_r1_filt = pred_r1[valid_pred_mask]
        pred_r2_filt = pred_r2[valid_pred_mask]
        # dones1_pred = dones1[valid_pred_mask]
        # dones2_pred = dones2[valid_pred_mask]
        truncated1_pred = truncated1[valid_pred_mask]
        truncated2_pred = truncated2[valid_pred_mask]
        # collisions1_pred = collisions1[valid_pred_mask]
        # collisions2_pred = collisions2[valid_pred_mask]
        # successes1_pred = successes1[valid_pred_mask]
        # successes2_pred = successes2[valid_pred_mask]
        goals1_pred = goals1[valid_pred_mask]
        goals2_pred = goals2[valid_pred_mask]
        imageobses1_pred = imageobses1[valid_pred_mask]
        imageobses2_pred = imageobses2[valid_pred_mask]

        # Call helper methods to compute labels.
        true_labels, seg_r1, seg_r2 = self.get_true_labels(true_r1_filt, true_r2_filt)
        pred_labels, seg_r1_pred, seg_r2_pred = self.get_pred_labels(pred_r1_filt, pred_r2_filt)
        llm_labels = self.get_llm_labels(obs1, obs2, acts1_pred, acts2_pred, pred_r1_filt, pred_r2_filt, pred_labels, true_labels, truncated1_pred, truncated2_pred, goals1_pred, goals2_pred, imageobses1_pred, imageobses2_pred, seg_r1, seg_r2, seg_r1_pred, seg_r2_pred)  
        
        # 8) Save to JSON Lines (always the same order for same seed!)
        record = {
            "true_labels": true_labels.tolist(),
            "pred_labels": pred_labels.tolist()
        }
        with open("enhanced_rl_records.jsonl", "a", encoding="utf-8") as fout:
            json.dump(record, fout, ensure_ascii=False)
            fout.write("\n")
        

        return obs1, obs2, true_r1, true_r2, pred_r1, pred_r2, true_labels, pred_labels, llm_labels


    def get_true_labels(self, r1, r2): # ✅
        print("Computing true labels...")
        
        seg_size = r1.shape[1]
        # Copy reward arrays to apply discounting.
        temp_r1 = r1.copy()
        temp_r2 = r2.copy()

        # Apply a decay factor (teacher_gamma) multiplicatively to earlier portions.
        for index in range(seg_size - 1):
            temp_r1[:, :index + 1] *= self.teacher_gamma
            temp_r2[:, :index + 1] *= self.teacher_gamma

        # Sum discounted rewards.
        segment_reward_r1 = temp_r1[:, -1] - temp_r1[:, 0]
        segment_reward_r2 = temp_r2[:, -1] - temp_r2[:, 0]

        # Identify ambiguous queries where the absolute difference is below a threshold.
        margin_index = np.abs(segment_reward_r1 - segment_reward_r2) < self.teacher_thres_equal

        # Rational labels: label 1 if first segment reward is less than second, else 0.
        rational_labels = (segment_reward_r1 < segment_reward_r2).astype(int)

        # Optionally refine labels using a Bradley-Terry style model.
        if self.teacher_beta > 0:
            # Convert to torch tensors.
            r_hat = torch.cat([
                torch.tensor(segment_reward_r1, dtype=torch.float32).unsqueeze(1),
                torch.tensor(segment_reward_r2, dtype=torch.float32).unsqueeze(1)
            ], dim=-1)
            r_hat = r_hat * self.teacher_beta
            prob_second = torch.nn.functional.softmax(r_hat, dim=-1)[:, 1]
            labels = torch.bernoulli(prob_second).int().numpy().reshape(-1, 1)
        else:
            labels = rational_labels.reshape(-1, 1)

        # Simulate teacher mistakes by randomly flipping labels.
        len_labels = labels.shape[0]
        rand_num = np.random.rand(len_labels)
        noise_index = rand_num <= self.teacher_eps_mistake
        labels[noise_index] = 1 - labels[noise_index]

        # Mark ambiguous queries as equally preferable with label -1.
        labels[margin_index.reshape(-1)] = -1

        # Save the computed labels as ground truth.
        gt_labels = labels.copy()
        print("Computed true labels successfully...")

        return gt_labels, segment_reward_r1, segment_reward_r2


    def get_pred_labels(self, r1, r2): # ✅
        print("Computing predicted labels...")

        seg_size = r1.shape[1]
        temp_r1 = r1.copy()
        temp_r2 = r2.copy()

        # Apply a decay factor (teacher_gamma) multiplicatively to earlier portions.
        for index in range(seg_size - 1):
            temp_r1[:, :index + 1] *= self.teacher_gamma
            temp_r2[:, :index + 1] *= self.teacher_gamma

        segment_reward_r1 = temp_r1[:, -1] - temp_r1[:, 0]
        segment_reward_r2 = temp_r2[:, -1] - temp_r2[:, 0]

        margin_index = np.abs(segment_reward_r1 - segment_reward_r2) < self.teacher_thres_equal

        rational_labels = (segment_reward_r1 < segment_reward_r2).astype(int)

        if self.teacher_beta > 0:
            r_hat = torch.cat([
                torch.tensor(segment_reward_r1, dtype=torch.float32).unsqueeze(1),
                torch.tensor(segment_reward_r2, dtype=torch.float32).unsqueeze(1)
            ], dim=-1)
            r_hat = r_hat * self.teacher_beta
            prob_second = torch.nn.functional.softmax(r_hat, dim=-1)[:, 1]
            labels = torch.bernoulli(prob_second).int().numpy().reshape(-1, 1)
        else:
            labels = rational_labels.reshape(-1, 1)

        len_labels = labels.shape[0]
        rand_num = np.random.rand(len_labels)
        noise_index = rand_num <= self.teacher_eps_mistake
        labels[noise_index] = 1 - labels[noise_index]

        labels[margin_index.reshape(-1)] = -1

        pred_labels = labels.copy()
        print("Computed predicted labels successfully...")

        return pred_labels, segment_reward_r1, segment_reward_r2


    # Save the computed LLM query accuracies.
    def convert_ndarray(self, obj): # ✅
        """
        Recursively convert all numpy arrays within the object to lists.
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self.convert_ndarray(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self.convert_ndarray(item) for item in obj]
        else:
            return obj


    def get_llm_labels(self, obs1, obs2, acts1, acts2, r1, r2, pred_labels, true_labels, truncated1, truncated2, goals1, goals2, images1, images2, sum_r1, sum_r2, sum_r1_pred, sum_r2_pred, use_rag=None):  # ✅

        print("Computing LLM labels...")
        # Save the current buffer index.
        origin_index = self.buffer_index
        use_rag = self.use_rag

        # Initialise an array for LLM labels, defaulting to -1 (invalid/ambiguous prediction).
        llm_labels = np.full(true_labels.shape, -1, dtype=int)
        true_sum_rewards_1 = np.zeros(true_labels.shape)
        true_sum_rewards_2 = np.zeros(true_labels.shape)
        pred_sum_rewards_1 = np.zeros(true_labels.shape)
        pred_sum_rewards_2 = np.zeros(true_labels.shape)

        double_check_fails = 0

        # Process each query pair (each corresponds to a multi-step trajectory segment).
        for idx in range(obs1.shape[0]):
            print("querying {} {}/{}".format(self.llm_model, idx+1, obs1.shape[0]))
            
            # Process the trajectory pair along with actions into a structured format.
            traj_1, traj_2, = traj_pair_process(
                obs1[idx],
                obs2[idx],
                acts1[idx],
                acts2[idx],
                # dones1[idx],
                # dones2[idx],
                truncated1[idx],
                truncated2[idx],
                # collisions1[idx],
                # collisions2[idx],
                # successes1[idx],
                # successes2[idx] 
                goals1[idx],
                goals2[idx],
                images1[idx],
                images2[idx]
            )
            image1_start_path, image1_end_path = traj_1["imageobses1"][0], traj_1["imageobses1"][-1] 
            image2_start_path, image2_end_path = traj_2["imageobses2"][0], traj_2["imageobses2"][-1]
            
            if use_rag:
                traj_str1, traj_str2, expert_actions_1_str, expert_actions_2_str = stringify_trajs(traj_1, traj_2, use_rag)
                # Call the GPT inference using the constructed prompt.
                answer = gpt_infer_rag(traj_str1, traj_str2, expert_actions_1_str, expert_actions_2_str, image1_start_path, image1_end_path, image2_start_path, image2_end_path, index=self.model_index)
            else:
                traj_str1, traj_str2 = stringify_trajs(traj_1, traj_2)
                # Call the GPT inference using the constructed prompt.
                answer = gpt_infer_no_rag(traj_str1, traj_str2, image1_start_path, image1_end_path, image2_start_path, image2_end_path, index=self.model_index)

            # Attempt to convert the answer into an integer label.
            try:
                label_res = int(answer)
                # Adjust labels: if the answer is 1 or 2, subtract 1 so that labels become 0 or 1.
                if label_res in [1, 2]:
                    label_res -= 1
                else:
                    label_res = -1
            except Exception:
                label_res = -1
            # print(f"{idx} LLM Answer ---> {label_res}")

            # Set default values for the swapped query.
            answer_swapped = None
            label_res_swapped = None
            # If double-checking is enabled and the first answer is valid, perform a swapped query.
            if self.double_check and label_res != -1:
                # Swap the order of the trajectories.
                if use_rag:
                    answer_swapped = gpt_infer_rag(traj_str2, traj_str1, expert_actions_2_str, expert_actions_1_str, image2_start_path, image2_end_path, image1_start_path, image1_end_path, index=self.model_index)
                else:
                    answer_swapped = gpt_infer_no_rag(traj_str2, traj_str1, image2_start_path, image2_end_path, image1_start_path, image1_end_path, index=self.model_index)
                try:
                    label_res_swapped = int(answer_swapped)
                    if label_res_swapped in [1, 2]:
                        label_res_swapped -= 1
                    else:
                        label_res_swapped = -1
                except Exception:
                    label_res_swapped = -1
                # print(f"{idx} Swapped LLM Answer ---> {label_res_swapped}")

                # If the two answers are not opposite/swapped, mark the label as invalid.
                if not ((label_res == 0 and label_res_swapped == 1) or 
                        (label_res == 1 and label_res_swapped == 0)):
                    print("Double check False!")
                    label_res = -1
                    double_check_fails += 1


            # Store the inferred label for this query.
            llm_labels[idx] = label_res
            true_sum_rewards_1[idx] = sum_r1[idx]
            true_sum_rewards_2[idx] = sum_r2[idx]
            pred_sum_rewards_1[idx] = sum_r1_pred[idx]
            pred_sum_rewards_2[idx] = sum_r2_pred[idx]

            data_to_save = {
                "index": idx,
                "traj1": traj_1,
                "traj2": traj_2,
                "llm_response": answer,
                "llm_response_swapped": answer_swapped,
                "label": label_res,
                "label_swapped": label_res_swapped,
                "double_check_fails": double_check_fails,
            }

            print("***************************************************************************************************")

            # Save the data to a JSONL file.
            with jsonlines.open("enhanced_rl/db/preferences/responses.jsonl", mode='a') as writer:
                writer.write(data_to_save)
            print("Saved segment to enhanced_rl/db/preferences/responses.jsonl")

        # After processing, compute two accuracy measures on valid predictions.
        valid_indices = [i for i, lab in enumerate(llm_labels) if lab != -1]
        if valid_indices:
            filtered_llm_labels = [llm_labels[i] for i in valid_indices]
            filtered_pred_labels = [pred_labels[i] for i in valid_indices]
            filtered_true_labels = [true_labels[i] for i in valid_indices]
            # Evaluate reward model accuracy
            correct_pred = sum(1 for pred, t in zip(filtered_pred_labels, filtered_true_labels) if pred == t)
            if len(filtered_pred_labels) > 0:
                accuracy_pred = correct_pred / len(filtered_pred_labels)
            else:
                accuracy_pred = 0
            # Evaluate LLM label accuracy
            correct_true = sum(1 for vlm_lab, t in zip(filtered_llm_labels, filtered_true_labels) if vlm_lab == t) 
            if len(filtered_llm_labels) > 0:
                accuracy_true = correct_true / len(filtered_llm_labels)
            else:
                accuracy_true = 0

            # Write each valid entry separately.
            with jsonlines.open("ablation/db/preferences/preferences.jsonl", mode='a') as writer:
                for i in valid_indices:
                    entry = {
                        "index": origin_index + i,
                        "llm_label": int(llm_labels[i]),
                        "prediction_label": int(pred_labels[i]),
                        "true_label": int(true_labels[i]),
                        "True_sum_r1": float(true_sum_rewards_1[i]),
                        "True_sum_r2": float(true_sum_rewards_2[i]),
                        "Pred_sum_r1": float(pred_sum_rewards_1[i]),
                        "Pred_sum_r2": float(pred_sum_rewards_2[i]),
                        "invalids": [i for i, lab in enumerate(llm_labels) if lab == -1],
                        "image1_start_path": image1_start_path,
                        "image1_end_path": image1_end_path,
                        "image2_start_path": image2_start_path,
                        "image2_end_path": image2_end_path
                    }
                    # Recursively convert any numpy arrays within the entry to lists.
                    entry = self.convert_ndarray(entry)
                    writer.write(entry)
                    print(f"Saved valid preference for index {origin_index + i} to ablation/db/preferences/preferences.jsonl")
        else:
            print("No valid predictions to compute accuracy.")
            accuracy_pred = accuracy_true = 0.0 

        # Save the computed LLM query accuracies.
        self.llm_query_accuracy_pred = accuracy_pred
        self.llm_query_accuracy_true = accuracy_true
        print("Computed LLM labels successfully...")

        return llm_labels


        # ****************************************************************************
    
    # 

    # --------------------------7. Training -----------------------------------

    def update(self):
        target_accuracy = 0.97
        current_accuracy = 0.0
        iteration = 0
        MAX_ITERATIONS = 1500

        save_step = self.save_step

        while current_accuracy < target_accuracy:
            ensemble_acc = self.train()
            
            # Determine the mean accuracy across ensemble members.
            current_accuracy = np.mean(ensemble_acc)
            
            iteration += 1
            print(f"[Update] Iteration {iteration} - Mean Ensemble Accuracy: {current_accuracy:.4f}")
            
            # Print target accuracy and additional info.
            if iteration % 10 == 0:
                print(f"[Update] Iteration {iteration}: Current mean accuracy {current_accuracy:.4f} vs target {target_accuracy}")
            
            # Safety break to prevent infinite loops.
            if iteration >= MAX_ITERATIONS:
                print("Maximum iterations reached; stopping training loop.")
                break

        print(f"Training stopped or converged with accuracy: {current_accuracy:.4f}")
        self.save(save_step=save_step)

        with open("enhanced_rl/db/preferences/trainresults.jsonl", "a") as f:
            json.dump({"step": save_step, "accuracy": current_accuracy}, f)
            f.write("\n")
            
        # self.KL_div.save()
        # print("Running Mean Std saved...")
            
    def get_threshold_beta(self):
        return max(self.threshold_beta_min, -(self.threshold_beta_init-self.threshold_beta_min)/self.k * self.update_step + self.threshold_beta_init)



    def train_with_denoise(self):
        print("Training reward model...")
        
        # Track loss and accuracy for each ensemble member
        ensemble_losses = [[] for _ in range(self.de)]
        ensemble_acc = np.array([0 for _ in range(self.de)])
        max_len = self.capacity if self.buffer_full else self.buffer_index
        print(f"max_len: {max_len}")
        
        # Log number of examples used for training
        num_examples = max_len
        print(f"[Train] Number of training examples: {num_examples}")  # 6
        
        # Convert buffers to numpy arrays for processing
        self.obs_buffer_seg1 = np.array(self.obs_buffer_seg1)
        self.obs_buffer_seg2 = np.array(self.obs_buffer_seg2)
        self.llm_labels_buffer = np.array(self.llm_labels_buffer)
        self.pred_labels_buffer = np.array(self.pred_labels_buffer)
        
        
        # === Compute Trustworthy Samples using Ensemble ===
        p_hat_all = []
        with torch.no_grad():
            for member in range(self.de):
                r_hat1 = self.r_hat_member_no_unsqueeze(self.obs_buffer_seg1[:max_len], member=member)
                r_hat2 = self.r_hat_member_no_unsqueeze(self.obs_buffer_seg2[:max_len], member=member)
                r_hat1 = r_hat1.sum(axis=1)
                r_hat2 = r_hat2.sum(axis=1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)  # (max_len, 2)
                print(f"r_hat shape: {r_hat.shape}")
                p_hat_all.append(F.softmax(r_hat, dim=-1).cpu())
        
        p_hat_all = torch.stack(p_hat_all)  # Shape: (de, max_len, 2)
        predict_probs = p_hat_all.mean(0)   # Ensemble average probabilities
        
        # Prepare LLM labels and mask out invalid ones
        llm_labels = torch.tensor(self.llm_labels_buffer[:max_len].flatten()).long()
        valid_mask = llm_labels >= 0
        preds_valid = predict_probs[valid_mask]
        labels_valid = llm_labels[valid_mask]
        
        # Create one-hot targets
        targets = torch.zeros_like(preds_valid)
        targets.scatter_(1, labels_valid.unsqueeze(1), 1)
        print(f"targets shape: {targets.shape}")
        
        # Compute KL divergence (cross entropy between predicted and LLM label)
        eps = 1e-12
        KL_div = - (targets * torch.log(preds_valid + eps)).sum(dim=1)
        
        # Compute mean loss
        print(f"KL_div shape: {KL_div.shape}")
        loss_per_sample = KL_div
        loss = loss_per_sample.mean()
        
        # === Filter Trustworthy Samples Based on KL Divergence ===
        x = KL_div.max().item()
        baseline = -np.log(x + 1e-8) + self.threshold_alpha * x
        uncertainty = min(self.get_threshold_beta() * KL_div.var().item(), 3.0)
        trust_sample_bool_index = KL_div < baseline + uncertainty
        trust_sample_index = np.where(trust_sample_bool_index.cpu().numpy())[0]
        
        # Update running stats with trusted KL samples
        self.KL_div.update(KL_div[trust_sample_bool_index].cpu().numpy())
        training_sample_index = trust_sample_index
        max_len = len(training_sample_index)
        
        # === Ensemble Training ===
        total_batch_index = [np.random.permutation(training_sample_index) for _ in range(self.de)]
        num_epochs = int(np.ceil(max_len / self.train_batch_size))
        total = 0  # total number of training samples (used for accuracy normalization)
        
        for epoch in range(num_epochs):
            loss = 0.0
            start = epoch * self.train_batch_size
            end = min((epoch + 1) * self.train_batch_size, max_len)
            
            self.opt.zero_grad()
            
            for member in range(self.de):
                idxs = total_batch_index[member][start:end]
                sa_t_1 = self.obs_buffer_seg1[idxs]
                sa_t_2 = self.obs_buffer_seg2[idxs]
                labels = self.pred_labels_buffer[idxs]
                labels = torch.from_numpy(labels.flatten()).long().to(self.device)
                
                if member == 0:
                    total += labels.size(0)
                
                # Get logits
                r1 = self.r_hat_member(sa_t_1, member=member).sum(axis=1)
                r2 = self.r_hat_member(sa_t_2, member=member).sum(axis=1)
                r = torch.cat([r1, r2], dim=-1)
                
                # Compute loss
                if self.label_margin > 0 or self.teacher_eps_equal > 0:
                    uniform_index = labels == -1
                    labels[uniform_index] = 0
                    target_onehot = torch.zeros_like(r).scatter(1, labels.unsqueeze(1), self.label_target)
                    target_onehot += self.label_margin
                    if uniform_index.int().sum().item() > 0:
                        target_onehot[uniform_index] = 0.5
                    curr_loss = self.softXEnt_loss(r, target_onehot)
                else:
                    curr_loss = self.CEloss(r, labels)
                
                loss += curr_loss
                ensemble_losses[member].append(curr_loss.item())
                
                # Accuracy
                _, predicted = torch.max(r.data, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct

            # Backprop once per batch (across ensemble members)
            loss.backward()
            self.opt.step()
        
        # Normalize accuracy
        ensemble_acc = ensemble_acc / total
        self.update_step += 1
        
        
        
        return ensemble_acc


    def train(self):
        print("Training reward model...")

        # Create lists for each ensemble member to store losses.
        ensemble_losses = [[] for _ in range(self.de)]
        ensemble_acc = np.zeros(self.de)
        gt_ensemble_acc = np.zeros(self.de)

        # Determine the number of training examples.
        num_examples = self.capacity if self.buffer_full else self.buffer_index
        print(f"[Train] Number of training examples: {num_examples}")

        # Convert the buffers to arrays.
        self.obs_buffer_seg1 = np.array(self.obs_buffer_seg1)
        self.obs_buffer_seg2 = np.array(self.obs_buffer_seg2)
        self.llm_labels_buffer = np.array(self.llm_labels_buffer)
        self.pred_labels_buffer = np.array(self.pred_labels_buffer)
        self.fake_flag = np.array(self.fake_flag)

        # Create a random permutation of indices for each ensemble member.
        total_batch_index = [np.random.permutation(num_examples) for _ in range(self.de)]

        # Compute the number of epochs required.
        num_epochs = int(np.ceil(num_examples / self.train_batch_size))

        total_samples = 0  # Count of overall examples used.
        gt_total = 0       # Count of valid examples.

        for epoch in range(num_epochs):
            # Iterate over each ensemble member.
            for member in range(self.de):
                self.opt.zero_grad()  # Reset gradients for the entire ensemble

                # Get the indices for the current batch.
                idxs = total_batch_index[member][epoch * self.train_batch_size : min((epoch + 1) * self.train_batch_size, num_examples)]
                sa_t_1 = self.obs_buffer_seg1[idxs]
                sa_t_2 = self.obs_buffer_seg2[idxs]
                llm_labels = self.llm_labels_buffer[idxs]
                fake_labels = self.fake_flag[idxs]

                # Convert LLM labels to PyTorch tensor.
                llm_labels_t = torch.from_numpy(np.array(llm_labels)).long().to(self.device)
                if llm_labels_t.dim() > 1 and llm_labels_t.size(1) == 1:
                    llm_labels_t = llm_labels_t.squeeze(1)

                # Use member 0 to update the sample counters.
                if member == 0:
                    total_samples += llm_labels_t.size(0)
                    gt_total += np.sum(np.array(fake_labels).flatten() == False)

                # Compute logits for each segment.
                r_hat1 = self.r_hat_batch_single_member(sa_t_1, member=member)
                r_hat2 = self.r_hat_batch_single_member(sa_t_2, member=member)

                # Stack predictions along a new dimension.
                logits = torch.stack([r_hat1, r_hat2], dim=1)  # Expected shape: (batch_size, 2)

                # Compute cross-entropy loss.
                curr_loss = self.CEloss(logits, llm_labels_t)

                ensemble_losses[member].append(curr_loss.item())

                # Get predictions and update accuracy.
                _, predicted = torch.max(logits, 1)
                ensemble_acc[member] += (predicted == llm_labels_t).sum().item()

                # Only count accuracy on valid samples (filtering out label -1).
                valid_mask = llm_labels_t != -1
                if valid_mask.any():
                    gt_correct = (predicted[valid_mask] == llm_labels_t[valid_mask]).sum().item()
                    gt_ensemble_acc[member] += gt_correct

                # Perform backward pass and optimisation for each member.
                curr_loss.backward()
                self.opt.step()  # Update parameters for the entire ensemble.

        # Normalise ensemble accuracies after all batches are processed.
        ensemble_acc = ensemble_acc / total_samples if total_samples > 0 else ensemble_acc
        gt_ensemble_acc = (gt_ensemble_acc / gt_total) if gt_total > 0 else gt_ensemble_acc

        print(f"[Train] Ensemble Accuracies: {ensemble_acc}")
        print(f"[Train] Ground-Truth Ensemble Accuracies: {gt_ensemble_acc}")

        # Optionally, print average loss per ensemble member.
        for m in range(self.de):
            avg_loss = np.mean(ensemble_losses[m]) if ensemble_losses[m] else float('nan')
            print(f"[Train] Ensemble Member {m} Average Loss: {avg_loss:.4f}")

        return ensemble_acc

    

    # --------------------------8. Sampling -----------------------------------

    # Instead of using uncertainty or diversity measures, it simply randomly samples queries with a batch size equal to mb_size. - Randomly selects queries without any additional criteria.
    def sample_queries(self, samp_index=1):  # ✅
        # Get queries from the buffer.
        if samp_index == 1:  # Use uniform sampling methods
            observations1, observations2, actions1, actions2, true_rewards1, true_rewards2, predicted_rewards1, predicted_rewards2,truncated1, truncated2, goals1, goals2, imageobses1, imageobses2 = self.get_queries_uniform()
        elif samp_index == 2:  # Use entropy sampling
            observations1, observations2, actions1, actions2, true_rewards1, true_rewards2, predicted_rewards1, predicted_rewards2, truncated1, truncated2, goals1, goals2, imageobses1, imageobses2 = self.get_queries_entropy()
        # elif samp_index == 3:  # Use KCenter sampling
        #     observations1, observations2, actions1, actions2, true_rewards1, true_rewards2, predicted_rewards1, predicted_rewards2, truncated1, truncated2, goals1, goals2, imageobses1, imageobses2 = self.get_queries_kcenter()
        # elif samp_index == 4:  # Use Disagreement sampling
        #     observations1, observations2, actions1, actions2, true_rewards1, true_rewards2, predicted_rewards1, predicted_rewards2, truncated1, truncated2, goals1, goals2, imageobses1, imageobses2 = self.disagreement_sampling()
        # elif samp_index == 5:  # Use KCenter Entropy sampling
        #     observations1, observations2, actions1, actions2, true_rewards1, true_rewards2, predicted_rewards1, predicted_rewards2, truncated1, truncated2, goals1, goals2, imageobses1, imageobses2 = self.kcenter_entropy_sampling()
        # elif samp_index == 6:  # Use KCenter Disagreement sampling
        #     observations1, observations2, actions1, actions2, true_rewards1, true_rewards2, predicted_rewards1, predicted_rewards2, truncated1, truncated2, goals1, goals2, imageobses1, imageobses2 = self.kcenter_disagreement_sampling()

    #     # Compute labels for each query pair and save in a preferences db.
        self.get_label(observations1, observations2, actions1, actions2, true_rewards1, true_rewards2, predicted_rewards1, predicted_rewards2, truncated1, truncated2, goals1, goals2, imageobses1, imageobses2)
        
        # # Load preferences from the preference buffer and use them to train the reward model.
        # num_preferences = self.put_queries()
        # if num_preferences:
        #     if num_preferences > 0:
        #         self.update()
        # else:
        #     print("No preferences found.")
        
            
        
    
    
    # ---------------------- enhanced_rl ----------------------------------------
    def relabel_with_predictor(
        self,
        replay_buffer,
        batch_size: int = 128,
        epochs: int = 500,
    ):
        device = self.device
        mse_loss = torch.nn.MSELoss()
        optimiser = self.opt

        max_idx = replay_buffer.buffer_size if replay_buffer.full else replay_buffer.pos
        if max_idx == 0:
            print("[RewardModel.enhanced_rl] Buffer is empty, nothing to enhanced_rl.")
            return False

        # Flatten observations and intrinsic rewards
        obs_shape = replay_buffer.observations.shape[2:]
        flat_obs = (
            replay_buffer.observations[:max_idx]
                .swapaxes(0, 1)                      # → (n_envs, steps, *obs_shape)
                .reshape(-1, *obs_shape)             # → (n_envs*steps, *obs_shape)
        )
        flat_intrinsic = (
            replay_buffer.rewards[:max_idx]
                .swapaxes(0, 1)                      # → (n_envs, steps)
                .reshape(-1)                         # → (n_envs*steps,)
        )
        flat_intrinsic = flat_intrinsic.astype(np.float32).reshape(-1)
        assert flat_intrinsic.ndim == 1, f"Expected 1D array, got shape {flat_intrinsic.shape}"

        total = flat_obs.shape[0]
        print(f"[RewardModel.enhanced_rl] {total} transitions → enhanced_rling for {epochs} epochs")
        target_tensor = torch.from_numpy(flat_intrinsic).to(device)

        # Move targets to tensor once
        target_tensor = torch.from_numpy(flat_intrinsic).float().to(device)

        for epoch in range(epochs):
            # shuffle indices
            perm = torch.randperm(total, device=device)
            for start in range(0, total, batch_size):
                end = min(start + batch_size, total)
                idxs = perm[start:end]

                # build obs batch
                obs_batch_np = flat_obs[idxs.cpu().numpy()]
                obs_batch = torch.from_numpy(obs_batch_np).float().to(device)

                # forward pass
                preds = self.r_hat_batch(obs_batch)  # should be (batch_size,) tensor

                # compute loss
                print(f"******* shape *********")
                print(preds.view(-1).shape, target_tensor[idxs].shape)
                loss = mse_loss(preds.view(-1), target_tensor[idxs])

                # backward + step
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

            print(f"[enhanced_rl] Epoch {epoch+1}/{epochs} done, last loss = {loss.item():.6f}")

        print("[RewardModel.enhanced_rl] Finished.")
        self.save(save_step=4800)

        return True

        
        

if __name__ == "__main__":
    load_step = 4800 # FOR LOADING increment this after training

    parser = argparse.ArgumentParser(description="Accept a sample index parameter from the command line.")
    parser.add_argument("--samp_index", type=int, choices=range(1, 7), required=True, help="Sample index must be one of {1, 2, 3, 4, 5, 6}")
    args = parser.parse_args()

    reward_model = RewardModel(ds=7, da=3)

    loaded_files = reward_model.load(load_step=4800)
    if not loaded_files:
        print("No pre-trained model found. A new model will be trained.")

    is_data = reward_model.add_data()
    if is_data:
        reward_model.sample_queries(samp_index=args.samp_index)