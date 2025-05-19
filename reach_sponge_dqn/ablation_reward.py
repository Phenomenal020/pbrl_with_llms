# sources: 
# RL-SaLLM-F paper: https://github.com/TU2021/RL-SaLLM-F/blob/main/reward_model.py
# Docstrings are automatically generated using codeium but verified manually for correctness

import numpy as np   # ✅

import torch  # ✅
import torch.nn as nn  # ✅
import torch.nn.functional as F  # ✅
import torch.optim as optim  # ✅

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

from api_ablation import (
    traj_pair_process,  stringify_trajs, gpt_infer_no_rag, gpt_infer_rag, get_image_paths
) # ✅


device = "cuda" if torch.cuda.is_available() else "cpu" # use cuda if available  # ✅

os.makedirs("ablation/llm/responses", exist_ok=True)
os.makedirs("ablation/llm/labels", exist_ok=True)
os.makedirs("ablation/llm/preferences", exist_ok=True)
os.makedirs("ablation/rm/", exist_ok=True)


# -----------------------1. Utility Functions -----------------------

def gen_net(in_size=1, out_size=1, H=256, n_layers=3, activation='relu',
            use_batch_norm=False, use_dropout=False, dropout_prob=0.3):
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
            mb_size = 50,      # mini-batch size used during ablation. ✅#
            activation='relu',  # activation function to use in the outer layer of the networks ✅
            capacity=1e5,       # total capacity of the preference buffer that stores training transitions ✅
            size_segment=5,    # length of each trajectory segment ✅
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
        self.use_rag = True
        self.model_index = 2

        self.vlm_label = vlm_label # ✅
        self.better_traj_gen = better_traj_gen # ✅
        self.double_check = double_check # ✅
        self.save_equal = save_equal # ✅
        self.vlm_feedback = vlm_feedback # ✅
        self.generate_check = generate_check # ✅
        self.capacity = int(capacity) # ✅

        # self.model_index = 1
        # self.index_update = True
        # self.model_index = 2
      
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
        self.dones = [] # ✅
        self.truncateds = [] # ✅
        self.collisions = [] # ✅
        self.successes = [] # ✅
        # self.goals = [] # ✅
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
        
        self.save_step = None # ✅
        self.add_index = 0 # ✅
        
        self.llm_model = None
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


    def add_data(self):
        file_paths = [
            "ablation/rm/trajectories/enhanced_reach_sponge3000132k.jsonl",
            "ablation/rm/trajectories/enhanced_reach_sponge3001132k.jsonl",
            "ablation/rm/trajectories/enhanced_reach_sponge3002132k.jsonl",
            "ablation/rm/trajectories/enhanced_reach_sponge3003132k.jsonl",
            "ablation/rm/trajectories/enhanced_reach_sponge3004132k.jsonl",
            "ablation/rm/trajectories/enhanced_reach_sponge3005132k.jsonl",
            "ablation/rm/trajectories/enhanced_reach_sponge3006132k.jsonl",
            "ablation/rm/trajectories/enhanced_reach_sponge3007132k.jsonl",  
            "ablation/rm/trajectories/enhanced_reach_sponge3008132k.jsonl",
            "ablation/rm/trajectories/enhanced_reach_sponge3009132k.jsonl",
            "ablation/rm/trajectories/enhanced_reach_sponge3010132k.jsonl",
            "ablation/rm/trajectories/enhanced_reach_sponge3011132k.jsonl",
            "ablation/rm/trajectories/enhanced_reach_sponge3012132k.jsonl",
            "ablation/rm/trajectories/enhanced_reach_sponge3013132k.jsonl",
            "ablation/rm/trajectories/enhanced_reach_sponge3014132k.jsonl",
            "ablation/rm/trajectories/enhanced_reach_sponge3015132k.jsonl"
        ]
        self.size_segment = 10
        print(f"Loading trajectory segments from JSONL files in segments of size {self.size_segment}...")

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
                            self.dones.pop(0)
                            self.truncateds.pop(0)
                            self.collisions.pop(0)
                            self.successes.pop(0)
                            self.imageobses.pop(0)
                        
                        # Append the new trajectory data.
                        self.obses.append(data["obs"][-self.size_segment:])
                        self.actions.append(data["act"][-self.size_segment:])
                        self.true_rewards.append(data["true_rew"][-self.size_segment:])
                        self.pred_rewards.append(data["pred_rew"][-self.size_segment:])
                        self.dones.append(data["done"][-self.size_segment:])
                        self.truncateds.append(data["truncated"][-self.size_segment:])
                        self.collisions.append(data["collision"][-self.size_segment:])
                        self.successes.append(data["success"][-self.size_segment:])
                        self.imageobses.append(data["image"][-self.size_segment:])
                        
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
        rng = np.random.default_rng(42)

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
        dones1 = np.array(self.dones)[indices1]
        truncated1 = np.array(self.truncateds)[indices1]
        collisions1 = np.array(self.collisions)[indices1]
        successes1 = np.array(self.successes)[indices1]
        # # goals1 = np.array(self.goals)[indices1]
        imageobses1 = np.array(self.imageobses)[indices1]

        obs2 = np.array(self.obses)[indices2]
        acts2 = np.array(self.actions)[indices2]
        true_r2 = np.array(self.true_rewards)[indices2]
        pred_r2 = np.array(self.pred_rewards)[indices2]
        dones2 = np.array(self.dones)[indices2]
        truncated2 = np.array(self.truncateds)[indices2]
        collisions2 = np.array(self.collisions)[indices2]
        successes2 = np.array(self.successes)[indices2]
        # # goals2 = np.array(self.goals)[indices2]
        imageobses2 = np.array(self.imageobses)[indices2]

        print("Got queries successfully using uniform sampling..")
        
        print(f"obs1: {obs1.shape}")
        print(f"obs2: {obs2.shape}")

        return (obs1, obs2, 
                acts1, acts2, 
                true_r1, true_r2, 
                pred_r1, pred_r2, 
                dones1, dones2, 
                truncated1, truncated2, 
                collisions1, collisions2, 
                successes1, successes2,
                # # goals1, goals2,
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
        dones1, dones2 = _slice(self.dones, idx1), _slice(self.dones, idx2)
        trunc1, trunc2 = _slice(self.truncateds, idx1), _slice(self.truncateds, idx2)
        coll1, coll2 = _slice(self.collisions, idx1), _slice(self.collisions, idx2)
        succ1, succ2 = _slice(self.successes, idx1), _slice(self.successes, idx2)   
        # # # # goals1, goals2 = _slice(self.goals, idx1), _slice(self.goals, idx2)  
        imageobses1, imageobses2 = _slice(self.imageobses, idx1), _slice(self.imageobses, idx2)  
        
        print("Got queries successfully via entropy sampling.")
        print(f"obs1: {obs1.shape}")
        print(f"obs2: {obs2.shape}")
        

        return (
            obs1, obs2,
            acts1, acts2,
            true_r1, true_r2,
            pred_r1, pred_r2,
            dones1, dones2,
            trunc1, trunc2,
            coll1, coll2,
            succ1, succ2,
            # # goals1, goals2,
            imageobses1, imageobses2
        )



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

        print("Stored queries successfully...")
        print(f"Buffer index: {self.buffer_index} / {capacity} (full={self.buffer_full})")

        return num_samples



    # assign labels (or preferences) to a pair of trajectory segments based on their predicted rewards   
    def get_label(self, obs1, obs2, acts1, acts2, true_r1, true_r2, pred_r1, pred_r2, dones1, dones2, truncated1, truncated2, collisions1, collisions2, successes1, successes2, imageobses1, imageobses2):  # ✅
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
        dones1_true = dones1[valid_true_mask]
        dones2_true = dones2[valid_true_mask]
        truncated1_true = truncated1[valid_true_mask]
        truncated2_true = truncated2[valid_true_mask]
        collisions1_true = collisions1[valid_true_mask]
        collisions2_true = collisions2[valid_true_mask]
        successes1_true = successes1[valid_true_mask]
        successes2_true = successes2[valid_true_mask]
        # # goals1_true = goals1[valid_true_mask]
        # # goals2_true = goals2[valid_true_mask]
        imageobses1_true = imageobses1[valid_true_mask]
        imageobses2_true = imageobses2[valid_true_mask]

        obs1_pred = obs1[valid_pred_mask]
        obs2_pred = obs2[valid_pred_mask]
        acts1_pred = acts1[valid_pred_mask]
        acts2_pred = acts2[valid_pred_mask]
        pred_r1_filt = pred_r1[valid_pred_mask]
        pred_r2_filt = pred_r2[valid_pred_mask]
        dones1_pred = dones1[valid_pred_mask]
        dones2_pred = dones2[valid_pred_mask]
        truncated1_pred = truncated1[valid_pred_mask]
        truncated2_pred = truncated2[valid_pred_mask]
        collisions1_pred = collisions1[valid_pred_mask]
        collisions2_pred = collisions2[valid_pred_mask]
        successes1_pred = successes1[valid_pred_mask]
        successes2_pred = successes2[valid_pred_mask]
        # # goals1_pred = goals1[valid_pred_mask]
        # # goals2_pred = goals2[valid_pred_mask]
        imageobses1_pred = imageobses1[valid_pred_mask]
        imageobses2_pred = imageobses2[valid_pred_mask]

        # Call helper methods to compute labels.
        true_labels, seg_r1, seg_r2 = self.get_true_labels(true_r1_filt, true_r2_filt)
        pred_labels, seg_r1_pred, seg_r2_pred = self.get_pred_labels(pred_r1_filt, pred_r2_filt)
        # llm_labels = self.get_llm_labels(obs1, obs2, acts1_pred, acts2_pred, pred_r1_filt, pred_r2_filt, pred_labels, true_labels, dones1_pred, dones2_pred, truncated1_pred, truncated2_pred, collisions1_pred, collisions2_pred, successes1_pred, successes2_pred, imageobses1_pred, imageobses2_pred, seg_r1, seg_r2, seg_r1_pred, seg_r2_pred)  
        
        # 8) Save to JSON Lines (always the same order for same seed!)
        record = {
            "true_labels": true_labels.tolist(),
            "pred_labels": pred_labels.tolist()
        }
        with open("ablation/rm/records.jsonl", "a", encoding="utf-8") as fout:
            json.dump(record, fout, ensure_ascii=False)
            fout.write("\n")
        

        # return obs1, obs2, true_r1, true_r2, pred_r1, pred_r2, true_labels, pred_labels, llm_labels
        return obs1, obs2, true_r1, true_r2, pred_r1, pred_r2, true_labels, pred_labels



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


    def get_llm_labels(self, obs1, obs2, acts1, acts2, r1, r2, pred_labels, true_labels, dones1, dones2, truncated1, truncated2, collisions1, collisions2, successes1, successes2,images1, images2, sum_r1, sum_r2, sum_r1_pred, sum_r2_pred, use_rag=None):  # ✅

        print("Computing LLM labels...")
        # Save the current buffer index.
        origin_index = self.buffer_index
        use_rag=self.use_rag

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
                dones1[idx],
                dones2[idx],
                truncated1[idx],
                truncated2[idx],
                collisions1[idx],
                collisions2[idx],
                successes1[idx],
                successes2[idx], 
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
            with jsonlines.open("ablation/llm/responses.jsonl", mode='a') as writer:
                writer.write(data_to_save)
            print("Saved segment to ablation/llm/responses.jsonl")

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
            with jsonlines.open("ablation/llm/preferences.jsonl", mode='a') as writer:
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
                        "observations1": obs1[i] if isinstance(obs1[i], list) else obs1[i].tolist(),
                        "observations2": obs2[i] if isinstance(obs2[i], list) else obs2[i].tolist(),
                        "invalids": [i for i, lab in enumerate(llm_labels) if lab == -1]
                    }
                    # Recursively convert any numpy arrays within the entry to lists.
                    entry = self.convert_ndarray(entry)
                    writer.write(entry)
                    print(f"Saved valid preference for index {origin_index + i} to ablation/llm/preferences.jsonl")
        else:
            print("No valid predictions to compute accuracy.")
            accuracy_pred = accuracy_true = 0.0 

        # Save the computed LLM query accuracies.
        self.llm_query_accuracy_pred = accuracy_pred
        self.llm_query_accuracy_true = accuracy_true
        print("Computed LLM labels successfully...")

        return llm_labels


        # ****************************************************************************


    def sample_queries(self, samp_index=1):  # ✅
        # Get queries from the buffer.
        if samp_index == 1:  # Use uniform sampling methods
            observations1, observations2, actions1, actions2, true_rewards1, true_rewards2, predicted_rewards1, predicted_rewards2, dones1, dones2, truncated1, truncated2, collisions1, collisions2, successes1, successes2,  imageobses1, imageobses2 = self.get_queries_uniform()
        elif samp_index == 2:  # Use entropy sampling
            observations1, observations2, actions1, actions2, true_rewards1, true_rewards2, predicted_rewards1, predicted_rewards2, dones1, dones2, truncated1, truncated2, collisions1, collisions2, successes1, successes2,  imageobses1, imageobses2= self.get_queries_entropy()

    #     # Compute labels for each query pair and save in a preferences db.
        self.get_label(observations1, observations2, actions1, actions2, true_rewards1, true_rewards2, predicted_rewards1, predicted_rewards2, dones1, dones2, truncated1, truncated2, collisions1, collisions2, successes1, successes2, imageobses1, imageobses2)
        
        # Load preferences from the preference buffer and use them to train the reward model.
        # num_preferences = self.put_queries()
        # if num_preferences:
        #     if num_preferences > 0:
        #         self.update()
        # else:
        #     print("No preferences found.")
        

if __name__ == "__main__":
    load_step = 128000 # FOR LOADING increment this after training

    parser = argparse.ArgumentParser(description="Accept a sample index parameter from the command line.")
    parser.add_argument("--samp_index", type=int, choices=range(1, 3), required=True, help="Sample index must be one of {1, 2}")
    args = parser.parse_args()

    reward_model = RewardModel(ds=7, da=3)
    
    print(reward_model.r_hat(np.array([2,2,2,2,2,2,2])))

    loaded_files = reward_model.load(load_step=128000)
    if not loaded_files:
        print("No pre-trained model found. A new model will be trained.")
        
    print(reward_model.r_hat(np.array([2,2,2,2,2,2,2])))

    is_data = reward_model.add_data()
    if is_data:
        reward_model.sample_queries(samp_index=args.samp_index)