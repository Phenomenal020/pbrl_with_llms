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

from api import (
    traj_pair_process,  stringify_trajs, gpt_infer_no_rag, gpt_infer_rag
) # ✅

from logger import reward_logger

# logger for get_queries method
queries_logger = reward_logger.queries_logger
# logger for add_data method
data_logger = reward_logger.data_logger
# logger for training method
train_logger = reward_logger.train_logger
# logger for put queries method
put_logger = reward_logger.put_logger
# logger for get true labels method
true_labels_logger = reward_logger.true_labels_logger
# logger for get pred labels method
pred_labels_logger = reward_logger.pred_labels_logger
# logger for get llm labels method
llm_labels_logger = reward_logger.llm_labels_logger
# logger for save model method 
save_model_logger = reward_logger.save_model_logger
# logger for load model method
load_model_logger = reward_logger.load_model_logger


device = "cuda" if torch.cuda.is_available() else "cpu" # use cuda if available  # ✅

os.makedirs("./db/ablation/responses", exist_ok=True)
os.makedirs("./db/ablation/labels", exist_ok=True)
os.makedirs("./db/preferences", exist_ok=True)


# -----------------------1. Utility Functions -----------------------

def gen_net(in_size=1, out_size=1, H=128, n_layers=3, activation='relu',
            use_batch_norm=False, use_dropout=False, dropout_prob=0.3):
    """
    Constructs a neural network with optional batch normalization and dropout.

    Args:
        in_size (int): Size of the input layer.
        out_size (int): Size of the output layer.
        H (int): Number of neurons in each hidden layer.
        n_layers (int): Number of hidden layers.
        activation (str): Output activation function ('tanh', 'sig', 'relu').
        use_batch_norm (bool): Whether to use batch normalization.
        use_dropout (bool): Whether to use dropout after activations.
        dropout_prob (float): Dropout probability.

    Returns:
        list: List of PyTorch layers representing the network.
    """

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


def KCenterGreedy(obs, full_obs, num_new_sample): # PASS
    pass


def compute_smallest_dist(obs, full_obs):  # PASS
    pass


class RewardModel:

    def __init__(self, #✅
            ds,                 # state dimension: size of the state/observation vector. ✅
            da,                 # action dimension: size of the action vector. ✅
            ensemble_size=3,    # number of reward predictors in the ensemble ✅
            lr=3e-4,            # learning rate for the optimiser ✅
            mb_size = 150,      # mini-batch size used during training. ✅#
            # mb_size = 50,      # mini-batch size used during ablation. ✅#
            activation='relu',  # activation function to use in the outer layer of the networks ✅
            capacity=1e5,       # total capacity of the preference buffer that stores training transitions ✅
            size_segment=10,    # length of each trajectory segment ✅
            max_size=800,       # maximum number of trajectory segments to keep (for FIFO logic) ✅  # Change to 10 to simulate (10x50) = 500 timesteps.
            large_batch=1,      # multiplier to increase the effective batch size during query sampling ✅
            label_margin=0.0,   # margin used when constructing soft target labels ✅
            teacher_beta=1,    # parameter for a Bradley-Terry model; if > 0, used to compute soft labels. For ablation, set to 0 to use hard rational labels instead. ✅
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
        self.size_segment = size_segment # ✅

        self.vlm_label = vlm_label # ✅
        self.better_traj_gen = better_traj_gen # ✅
        self.double_check = double_check # ✅
        self.save_equal = save_equal # ✅
        self.vlm_feedback = vlm_feedback # ✅
        self.generate_check = generate_check # ✅
        self.capacity = int(capacity) # ✅

        self.save_step = 5000 # ✅

        self.model_index = 1
        self.index_update = True
      
        if traj_action:
            self.obs_buffer_seg1 = np.empty((self.capacity, size_segment, self.ds+self.da), dtype=np.float32) # ✅
            self.obs_buffer_seg2 = np.empty((self.capacity, size_segment, self.ds+self.da), dtype=np.float32) # ✅
        else:
            self.obs_buffer_seg1 = np.empty((self.capacity, size_segment, self.ds), dtype=np.float32) # ✅
            self.obs_buffer_seg2 = np.empty((self.capacity, size_segment, self.ds), dtype=np.float32) # ✅
        self.pred_labels_buffer = np.empty((self.capacity, 1), dtype=np.float32) # ✅
        self.llm_labels_buffer = np.empty((self.capacity, 1), dtype=np.float32) # ✅
        self.true_labels_buffer = np.empty((self.capacity, 1), dtype=np.float32) # ✅

        self.buffer_env_id = np.empty((self.capacity, 1), dtype=np.int32)  # Stores environment ID to track which environment the trajectory was collected from ❌
        
        # fake_flag is a boolean array of shape (capacity, 1) that indicates whether a particular entry in the buffer is fake or not.
        # An entry is considered fake if it is generated by the model, rather than being a real trajectory collected from the environment.
        # The buffer_index is the current index in the buffer where the next trajectory segment will be stored.
        # The buffer_full flag is set to True if the buffer is full, i.e., if the buffer_index has reached the capacity of the buffer.
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

        self.mb_size = mb_size # ✅
        self.origin_mb_size = mb_size # ✅
        self.train_batch_size = 32  # ✅

        self.CEloss = nn.CrossEntropyLoss() # ✅

        self.running_means = [] # ❌
        self.running_stds = [] # ❌
        self.best_seg = [] # ❌
        self.best_label = [] # ❌
        self.best_action = [] # ❌
        self.large_batch = large_batch # ❌
        
        # new teacher
        self.teacher_beta = teacher_beta # ✅ 
        self.teacher_gamma = teacher_gamma # ✅
        self.teacher_eps_mistake = teacher_eps_mistake # ✅
        self.teacher_eps_equal = teacher_eps_equal # ✅
        self.teacher_eps_skip = teacher_eps_skip # ✅
        self.teacher_thres_skip = -1 # do not use reward difference to skip queries ✅
        self.teacher_thres_equal = 0 # ✅
        
        self.label_margin = label_margin # ❌
        self.label_target = 1 - 2*self.label_margin # ❌

        # for llms
        self.llm_query_accuracy = 0 # ✅
        self.llm_label_accuracy = 0 # ✅

        self.device = device # ✅

        print(f"Is temp buffer empty?: {len(self.obses) == 0}")

    
    def construct_ensemble(self): # ✅
        """
        Construct an ensemble of neural network models for reward prediction.
        This function initialises a specified number of neural network models (based on the value of `self.de`) and appends them to the ensemble. Each model is created using a generator function `gen_net` with specific input sizes depending on whether trajectory actions are considered. The models are added to the ensemble list, and their parameters are appended to a parameter list for optimisation. An Adam optimiser is then created over these combined parameters.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
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


    def softXEnt_loss(self, input, target): # ✅
        """
        Calculate the soft cross-entropy loss between input and target.

        Parameters
        ----------
        input : torch.tensor
            The input tensor of shape (batch_size, n_classes)
        target : torch.tensor
            The target tensor of shape (batch_size, n_classes)

        Returns
        -------
        loss : torch.tensor
            The soft cross-entropy loss of shape (1,)
        """
        logprobs = torch.nn.functional.log_softmax (input, dim = 1)
        return  -(target * logprobs).sum() / input.shape[0]
    

    # ------------------------- 3. Trajectory Data Storage --------------------------


    def add_data(self): # ✅
        """
        Loads trajectory segments from multiple JSON Lines files and adds them to the reward model's temporary buffers.
        Each line of the file is a JSON object with keys:
            "obs", "act", "true_rew", "pred_rew", "done", "truncated", "collision", "success".
        """
        file_paths = [
            "db/ablation/enhanced_reach_sponge5100.jsonl",
            "db/ablation/enhanced_reach_sponge5105.jsonl",
            "db/ablation/enhanced_reach_sponge5110.jsonl",
            "db/ablation/enhanced_reach_sponge5115.jsonl",
            "db/ablation/enhanced_reach_sponge5120.jsonl"
        ]  # For ablation
        
        print("Loading trajectory segments from JSONL files...")
        data_logger.info("Loading trajectory segments from JSONL files...")

        for file_path in file_paths:
            try:
                with open(file_path, "r") as f:
                    # Process each line in the file
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        
                        # Load the JSON object
                        data = json.loads(line)
                        
                        # # Log the data loading
                        # print(f"Loaded trajectory segment from {file_path}.")
                        # data_logger.info(f"Loaded trajectory segment from {file_path}.")
                        
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
                        
                        # Append the new trajectory data.
                        self.obses.append(data["obs"])
                        self.actions.append(data["act"])
                        self.true_rewards.append(data["true_rew"])
                        self.pred_rewards.append(data["pred_rew"])
                        self.dones.append(data["done"])
                        self.truncateds.append(data["truncated"])
                        self.collisions.append(data["collision"])
                        self.successes.append(data["success"])
                        
                    # Log that we finished processing a file.
                    data_logger.info(f"Added trajectory segments from {file_path} to RM buffers.")
                    # print(f"Finished loading file: {file_path}")
            
            except FileNotFoundError:
                error_msg = f"File not found: {file_path}"
                data_logger.error(error_msg)
                # print(error_msg)
            except json.JSONDecodeError as e:
                error_msg = f"Error decoding JSON in file {file_path}: {e}"
                data_logger.error(error_msg)
                # print(error_msg)
            except Exception as e:
                error_msg = f"An error occurred while reading {file_path}: {e}"
                data_logger.error(error_msg)
                # print(error_msg)

        print(f"Temporary buffer size: {len(self.obses)}")
        return True

    # --------------------------4. Probability and rewards --------------------------

    def get_rank_probability(self, x_1, x_2): # PASS
        pass
    
    
    def get_entropy(self, x_1, x_2): # PASS
       pass
    

    def p_hat_entropy(self, x_1, x_2, member=-1): # PASS
        pass


    def p_hat(self, x_1, x_2): # ✅
        """
        Compute the probability that x_1 is preferred to x_2.

        Parameters
        ----------
        x_1 : torch.Tensor
            The first input to compare.
        x_2 : torch.Tensor
            The second input to compare.

        Returns
        -------
        torch.Tensor
            The probability that x_1 is preferred to x_2.
        """
        with torch.no_grad():
            # Get the average reward predictions for x_1 and x_2.
            r_hat1 = self.r_hat(x_1)
            r_hat2 = self.r_hat(x_2)

            # Concatenate the predictions along the last dimension.
            r_hat_concat = torch.cat([r_hat1, r_hat2], dim=-1)
        
        # Apply softmax along the last dimension to get probabilities.
        probabilities = F.softmax(r_hat_concat, dim=-1)
        
        # Return the probability of x_1 being preferred (first column).
        return probabilities[:, 0]


    # Computes reward for a given input and ensemble member.
    # Called by r_hat and r_hat_batch self.de times
    def r_hat_member(self, x, member=-1): # ✅
        """
        Compute the reward prediction for a given input and ensemble member.

        Parameters
        ----------
        x : torch.Tensor or np.ndarray
            The input data for which the reward is to be predicted.
        member : int, optional
            The index of the ensemble member to use for prediction. Defaults to -1, which uses the last member.

        Returns
        -------
        torch.Tensor
            The predicted reward for the input data by the specified ensemble member.
        """
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


    # Computes the average reward for a given input and all ensemble members
    def r_hat(self, x): # ✅
        """
        Compute the average reward prediction from all ensemble members for a given input.

        Parameters
        ----------
        x : torch.Tensor or np.ndarray
            The input data for which the reward is to be predicted.

        Returns
        -------
        torch.Tensor
            The average predicted reward for the input data by all ensemble members.
        """
        r_hats = []
        for member in range(self.de):
            r_hats.append(self.r_hat_member(x, member=member))
        r_hats = torch.stack(r_hats, dim=0)
        return r_hats.mean(dim=0) / 10

    
    # Computes the average reward for a given batch of inputs and returns a vector of rewards corresponding to the batched mean reward for all members
    def r_hat_batch(self, x): # ✅
        """
        Compute the average reward prediction from all ensemble members for a given batch of input data.

        Parameters
        ----------
        x : torch.Tensor or np.ndarray
            The batched input data for which the reward is to be predicted.

        Returns
        -------
        np.ndarray
            The average predicted rewards for the input data by all ensemble members.
        """
        # If x is a NumPy array, convert it to a torch.Tensor.
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x)
        # Ensure input is on the correct device and type
        x = x.float().to(self.device)
        
        r_hats = []
        for member in range(self.de):
            # Compute reward for each ensemble member
            member_reward = self.r_hat_member(x, member=member)
            r_hats.append(member_reward.detach().cpu().numpy())
        
        r_hats = np.array(r_hats)  # shape: (de, batch_size)
        # Average across the ensemble members (axis=0)
        return np.mean(r_hats, axis=0)


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






    # --------------------------5. Checkpointing --------------------------

    # state_dict() Returns a dictionary containing the model's parameters. Save the model's state dictionary to disc - before rl_model.save()
    def save(self, model_dir="reward_model/enhanced_rl/ppo", save_step=0): # ✅
        """
        Save the state dictionaries of all ensemble members to disc

        Parameters
        ----------
        model_dir : str
            The directory where the model files will be saved.

        Returns
        -------
        list
            A list of file paths where models were saved.
        """
        print("Saving reward model...")
        save_model_logger.info("Saving reward model...")

        # Ensure directory exists. If not, create it.
        os.makedirs(model_dir, exist_ok=True)

        saved_files = []
        for member in range(self.de):
            file_path = os.path.join(model_dir, f"reward_model_{save_step}_{member}.pt")
            torch.save(self.ensemble[member].state_dict(), file_path)
            saved_files.append(file_path)

        print("Reward model saved successfully.")
        save_model_logger.info("Reward model saved successfully.")

        return saved_files  # Return list of saved file paths
    
    

    def load(self, model_dir="reward_model/enhanced_rl/ppo", load_step=0): # ✅
        """
        Load the state dictionaries of all ensemble members from disc.

        Parameters
        ----------
        model_dir : str
            The directory where the model files are saved.

        Returns
        -------
        list
            A list of file paths of successfully loaded models.
        """
        print("Loading reward model...")
        load_model_logger.info("Loading reward model...")

        if load_step != 25000:
            raise Exception("Reward model loading is only supported for step 10000.")

        loaded_files = []
        for member in range(self.de):
            file_path = os.path.join(model_dir, f"reward_model_{load_step}_{member}.pt")

            if not os.path.exists(file_path):
                print(f"Warning: Reward model file {file_path} not found.")
                load_model_logger.warning(f"Reward model file {file_path} not found.")
                continue  # Skip missing models instead of crashing

            with torch.no_grad():  # Prevent tracking gradients while loading
                self.ensemble[member].load_state_dict(torch.load(file_path))

            loaded_files.append(file_path)

        print("Reward model loaded successfully.")
        load_model_logger.info("Reward model loaded successfully.")

        return loaded_files  # Return list of loaded file paths
    


    # --------------------------6. Getting Labels --------------------------

    def sample_indices(self, sample_buffer, mb_size, samp_index=1, rng=None): # ✅
        """
        Generate indices for sampling segments from the sample_buffer.
        For now, supports only uniform sampling.
        
        Parameters:
            sample_buffer (list or array): Buffer containing segments.
            mb_size (int): Number of segments to sample.
            samp_index (int): Sampling strategy indicator (only 1 is implemented for now).
            rng (np.random.Generator): A random generator for reproducibility.
        
        Returns:
            indices (array): Array of sampled indices.
        """
        if rng is None:
            rng = np.random.default_rng()

        buffer = np.array(sample_buffer)

        # Verify that the buffer has at least one segment with the required length.
        if len(buffer) < 1 or (len(buffer[0]) < self.size_segment):
            raise ValueError(f"Sample buffer does not contain any segment of the required length {self.size_segment}")

        if samp_index == 1:
            # Uniform sampling: randomly choose indices with replacement.
            indices = rng.choice(len(buffer), size=mb_size, replace=True)
        else:
            raise ValueError(f"Invalid sampling method index: {samp_index}. Must be 1-5.")

        return indices


    def get_queries(self, mb_size=3, samp_index=1, seed=42): # ✅
        """
        If the trajectory arrays (obses, actions, etc.) are flat, sample mb_size segments
        from each one using common indices to ensure the segments correspond across arrays.
        A seed is provided so that the same two sets of segments are sampled across runs,
        and the second set is guaranteed to be different from the first set.
        
        Returns:
            tuple: A tuple of sampled queries corresponding to the various trajectory arrays.
        """
        print("Getting queries...")
        queries_logger.info("Getting queries...")

        # Instantiate a dedicated random generator with the provided seed.
        rng = np.random.default_rng(seed)

        # Check for available trajectory data.
        if len(self.obses) == 0:
            queries_logger.error("No trajectory data available.")
            raise ValueError("No trajectory data available.")

        # # Sample indices for the first set using the provided rng.
        indices1 = self.sample_indices(self.obses, mb_size, samp_index, rng=rng)
        obs1 = np.array(self.obses)[indices1]
        acts1 = np.array(self.actions)[indices1]
        true_r1 = np.array(self.true_rewards)[indices1]
        pred_r1 = np.array(self.pred_rewards)[indices1]
        dones1 = np.array(self.dones)[indices1]
        truncated1 = np.array(self.truncateds)[indices1]
        collisions1 = np.array(self.collisions)[indices1]
        successes1 = np.array(self.successes)[indices1]

        # Compute the set of remaining indices - not chosen in indices1.
        all_indices = np.arange(len(self.obses))
        remaining_indices = np.setdiff1d(all_indices, indices1)

        # Sample new indices for the second set from the remaining indices.
        if len(remaining_indices) < mb_size:
            # Not enough unique segments are available; sample with replacement.
            indices2 = rng.choice(remaining_indices, size=mb_size, replace=True)
        else:
            indices2 = rng.choice(remaining_indices, size=mb_size, replace=False)

        obs2 = np.array(self.obses)[indices2]
        acts2 = np.array(self.actions)[indices2]
        true_r2 = np.array(self.true_rewards)[indices2]
        pred_r2 = np.array(self.pred_rewards)[indices2]
        dones2 = np.array(self.dones)[indices2]
        truncated2 = np.array(self.truncateds)[indices2]
        collisions2 = np.array(self.collisions)[indices2]
        successes2 = np.array(self.successes)[indices2]

        queries_logger.info("Got queries successfully.")
        print("Got queries successfully.")

        return (obs1, obs2, 
                acts1, acts2, 
                true_r1, true_r2, 
                pred_r1, pred_r2, 
                dones1, dones2, 
                truncated1, truncated2, 
                collisions1, collisions2, 
                successes1, successes2)


    # This method stores the query pairs (and their labels) into circular buffers. It handles buffer overflow with a FIFO (First-In-First-Out) mechanism.
    def put_queries(self):  # ✅
        # get [obs1, obs2, pred labels, llm labels, true labels] from preferences buffer, D
        file_path = "./db/preferences/preferences.jsonl"
        try:
            with open(file_path, "r") as f:
                preferences = [json.loads(line) for line in f]
            print(f"Successfully read preferences.")  
        except FileNotFoundError:
            print("Preferences file not found.")
            return

        # Aggregate the data across all preference entries.
        all_obs1 = []
        all_obs2 = []
        all_pred_labels = []
        all_llm_labels = []
        all_true_labels = []

        for preference in preferences:
            # Each key already holds a list. Extend our aggregators with them.
            all_obs1.append(preference["observations1"])
            all_obs2.append(preference["observations2"])
            all_pred_labels.append(preference["prediction_label"])
            all_llm_labels.append(preference["llm_label"])
            all_true_labels.append(preference["true_label"])

        # Validate input shapes: ensure that all aggregated lists have the same number of samples.
        num_samples = len(all_obs1)
        print(f"Total samples aggregated: {num_samples}")

        # Convert aggregated lists to NumPy arrays for consistency in copy operations.
        all_obs1 = np.array(all_obs1)
        all_obs2 = np.array(all_obs2)
        all_pred_labels = np.array(all_pred_labels).reshape(-1, 1)
        all_llm_labels = np.array(all_llm_labels).reshape(-1, 1)
        all_true_labels = np.array(all_true_labels).reshape(-1, 1)

        # Determine where to insert the preferences in the buffer.
        next_index = self.buffer_index + num_samples

        # If this will cause the preferences buffer to overflow, wrap around.
        if next_index >= self.capacity:
            # Buffer will wrap around.
            self.buffer_full = True
            # Calculate remaining space in the buffer.
            remaining_space = self.capacity - self.buffer_index

            # Insert data into the remaining slots.
            np.copyto(self.obs_buffer_seg1[self.buffer_index:], all_obs1[:remaining_space])
            np.copyto(self.obs_buffer_seg2[self.buffer_index:], all_obs2[:remaining_space])
            np.copyto(self.pred_labels_buffer[self.buffer_index:], all_pred_labels[:remaining_space])
            np.copyto(self.llm_labels_buffer[self.buffer_index:], all_llm_labels[:remaining_space])
            np.copyto(self.true_labels_buffer[self.buffer_index:], all_true_labels[:remaining_space])

            # Wrap around: Insert remaining data at the beginning of the buffer.
            remaining_samples = num_samples - remaining_space
            if remaining_samples > 0:
                np.copyto(self.obs_buffer_seg1[:remaining_samples], all_obs1[remaining_space:])
                np.copyto(self.obs_buffer_seg2[:remaining_samples], all_obs2[remaining_space:])
                np.copyto(self.pred_labels_buffer[:remaining_samples], all_pred_labels[remaining_space:])
                np.copyto(self.llm_labels_buffer[:remaining_samples], all_llm_labels[remaining_space:])
                np.copyto(self.true_labels_buffer[:remaining_samples], all_true_labels[remaining_space:])

            # Update the buffer index to the new starting position.
            self.buffer_index = remaining_samples

        else:
            # There is enough space in the buffer; insert directly.
            np.copyto(self.obs_buffer_seg1[self.buffer_index:next_index], all_obs1)
            np.copyto(self.obs_buffer_seg2[self.buffer_index:next_index], all_obs2)
            np.copyto(self.pred_labels_buffer[self.buffer_index:next_index], all_pred_labels)
            np.copyto(self.llm_labels_buffer[self.buffer_index:next_index], all_llm_labels)
            np.copyto(self.true_labels_buffer[self.buffer_index:next_index], all_true_labels)

            # Move the buffer index forward.
            self.buffer_index = next_index

        print("Stored queries successfully...")
        put_logger.info("Stored queries successfully...")
        print(f"Buffer size: {self.buffer_index} === {self.obs_buffer_seg1.shape[0]}")

        return num_samples


    # assign labels (or preferences) to a pair of trajectory segments based on their predicted rewards   
    def get_label(self, obs1, obs2, acts1, acts2, true_r1, true_r2, pred_r1, pred_r2, dones1, dones2, truncated1, truncated2, collisions1, collisions2, successes1, successes2): # ✅
        """
        Compute labels for pairs of trajectory segments based on their rewards.
        Returns:
        obs1, obs2, true_r1, true_r2, pred_r1, pred_r2, dones1, dones2, truncated1, truncated2, collisions1, collisions2, successes1, successes2, true_labels, pred_labels, llm_labels
        """
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

        # Call helper methods to compute labels.
        true_labels, seg_r1, seg_r2 = self.get_true_labels(true_r1_filt, true_r2_filt)
        pred_labels, seg_r1_pred, seg_r2_pred = self.get_pred_labels(pred_r1_filt, pred_r2_filt)
        llm_labels = self.get_llm_labels(obs1, obs2, acts1_pred, acts2_pred, pred_r1_filt, pred_r2_filt, pred_labels, true_labels, dones1_pred, dones2_pred, truncated1_pred, truncated2_pred, collisions1_pred, collisions2_pred, successes1_pred, successes2_pred, seg_r1, seg_r2, seg_r1_pred, seg_r2_pred)   

        return obs1, obs2, true_r1, true_r2, pred_r1, pred_r2, true_labels, pred_labels, llm_labels


    def get_true_labels(self, r1, r2): # ✅
        """
        Compute discretised labels from true rewards.
        r1, r2: arrays of true rewards for segments, shape [n, seg_size]
        Returns an array of labels (with ambiguous queries marked as -1).
        """
        print("Computing true labels...")
        true_labels_logger.info("Computing true labels...")
        
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

        true_labels_logger.info("Computed true labels successfully...")
        true_labels_logger.debug(f"True Labels: {np.array_str(gt_labels)}")
        return gt_labels, segment_reward_r1, segment_reward_r2


    def get_pred_labels(self, r1, r2): # ✅
        """
        Compute discretized labels from predicted rewards.
        r1, r2: arrays of predicted rewards for segments, shape [n, seg_size]
        Returns an array of labels (with ambiguous queries marked as -1).
        """

        print("Computing predicted labels...")
        pred_labels_logger.info("Computing predicted labels...")

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

        pred_labels_logger.info("Computed predicted labels successfully...")
        pred_labels_logger.debug(f"Predicted Labels: {np.array_str(pred_labels)}")
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


    def get_llm_labels(self, obs1, obs2, acts1, acts2, r1, r2, pred_labels, true_labels, dones1, dones2, truncated1, truncated2, collisions1, collisions2, successes1, successes2, sum_r1, sum_r2, sum_r1_pred, sum_r2_pred, use_rag=True):  # ✅

        print("Computing LLM labels...")
        llm_labels_logger.info("Computing LLM labels...")

        # Save the current buffer index.
        origin_index = self.buffer_index

        # Initialise an array for LLM labels, defaulting to -1 (invalid/ambiguous prediction).
        llm_labels = np.full(true_labels.shape, -1, dtype=int)
        true_sum_rewards_1 = np.zeros(true_labels.shape)
        true_sum_rewards_2 = np.zeros(true_labels.shape)
        pred_sum_rewards_1 = np.zeros(true_labels.shape)
        pred_sum_rewards_2 = np.zeros(true_labels.shape)

        double_check_fails = 0

        # Process each query pair (each corresponds to a multi-step trajectory segment).
        for idx in range(obs1.shape[0]):
            print("querying {} {}/{}".format(self.vlm_label, idx, obs1.shape[0]))
            
            # Process the trajectory pair along with actions into a structured format.
            traj_1, traj_2 = traj_pair_process(
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
                successes2[idx] 
            )
            
            if use_rag:
                traj_str1, traj_str2, expert_actions_1_str, expert_actions_2_str = stringify_trajs(traj_1, traj_2, use_rag)
                # Call the GPT inference using the constructed prompt.
                answer = gpt_infer_rag(traj_str1, traj_str2, expert_actions_1_str, expert_actions_2_str, index=self.model_index)
            else:
                traj_str1, traj_str2 = stringify_trajs(traj_1, traj_2)
                # Call the GPT inference using the constructed prompt.
                answer = gpt_infer_no_rag(traj_str1, traj_str2, index=self.model_index)

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
                    answer_swapped = gpt_infer_rag(traj_str2, traj_str1, expert_actions_2_str, expert_actions_1_str, index=self.model_index)
                else:
                    answer_swapped = gpt_infer_no_rag(traj_str2, traj_str1, index=self.model_index)
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

            self.model_index += 1

            print("***************************************************************************************************")

            # Save the data to a JSONL file.
            with jsonlines.open("./db/preferences/responses.jsonl", mode='a') as writer:
                writer.write(data_to_save)
            print("Saved segment to ./db/preferences/responses.jsonl")

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
            llm_labels_logger.info(f"RM Accuracy (Predicted): {accuracy_pred * 100:.2f}%")
            llm_labels_logger.info(f"LLM Label Accuracy (True): {accuracy_true * 100:.2f}%")

            # Write each valid entry separately.
            with jsonlines.open("./db/preferences/preferences.jsonl", mode='a') as writer:
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
                    }
                    # Recursively convert any numpy arrays within the entry to lists.
                    entry = self.convert_ndarray(entry)
                    writer.write(entry)
                    print(f"Saved valid preference for index {origin_index + i} to ./db/preferences/preferences.jsonl")
        else:
            print("No valid predictions to compute accuracy.")
            llm_labels_logger.info("No valid predictions to compute accuracy.")
            accuracy_pred = accuracy_true = 0.0 

        # Save the computed LLM query accuracies.
        self.llm_query_accuracy_pred = accuracy_pred
        self.llm_query_accuracy_true = accuracy_true
        print("Computed LLM labels successfully...")
        llm_labels_logger.info("Computed LLM labels successfully...")

        return llm_labels


        # ****************************************************************************
    
    

    # --------------------------7. Training -----------------------------------

    def update(self):
        target_accuracy = 0.96
        current_accuracy = 0.0
        iteration = 0
        MAX_ITERATIONS = 400

        save_step = 25000

        while current_accuracy < target_accuracy:
            # Call the train method which returns (llm_query_accuracy, llm_label_accuracy, ensemble_acc)
            llm_query_accuracy, llm_label_accuracy, ensemble_acc = self.train()
            
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

        with open("./db/preferences/trainresults.jsonl", "a") as f:
            json.dump({"step": save_step, "accuracy": current_accuracy}, f)
            f.write("\n")



    def train(self):
        """
        Train the reward model using hard targets (cross-entropy loss)
        """
        print("Training reward model...")
        train_logger.info("Training reward model...")

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

        return self.llm_query_accuracy, self.llm_label_accuracy, ensemble_acc




    

    # --------------------------8. Sampling -----------------------------------

    # Instead of using uncertainty or diversity measures, it simply randomly samples queries with a batch size equal to mb_size. - Randomly selects queries without any additional criteria.
    def uniform_sample(self, samp_index=1):  # ✅
        # # Get queries from the buffer.
        # observations1, observations2, actions1, actions2, true_rewards1, true_rewards2, predicted_rewards1, predicted_rewards2, dones1, dones2, truncated1, truncated2, collisions1, collisions2, successes1, successes2 = self.get_queries(mb_size=self.mb_size, samp_index=samp_index)

        # # Compute labels for each query pair and save in a preferences db.
        # self.get_label(observations1, observations2, actions1, actions2, true_rewards1, true_rewards2, predicted_rewards1, predicted_rewards2, dones1, dones2, truncated1, truncated2, collisions1, collisions2, successes1, successes2)
        
        # Load preferences from the preference buffer and use them to train the reward model.
        num_preferences = self.put_queries()
        if num_preferences > 0:
            self.update()

    

    # When implemented fully, its goal would be to return a number of selected queries (or their labels) that have been chosen solely based on diversity using the k-center approach
    def kcenter_sampling(self): # PASS
        pass
    

    # The k-center-based methods (including the variants that incorporate disagreement or entropy) aim to choose a diverse subset of queries. This helps ensure that the training data covers different regions of the input space.
    def kcenter_disagree_sampling(self): # PASS
        pass
    

    # Similar to kcenter_disagree_sampling, except that it uses entropy instead of disagreement from the ensemble.
    def kcenter_entropy_sampling(self): # PASS
        pass

   

    # These methods select queries where the ensemble is uncertain (i.e. high disagreement or high entropy). They are useful if you want the model to focus on samples where it is unsure, which can be very informative for training.
    def disagreement_sampling(self): # PASS
        pass
    

    def entropy_sampling(self): # PASS
        pass




if __name__ == "__main__":
    load_step = 25000 # FOR LOADING increment this after training

    parser = argparse.ArgumentParser(description="Accept a sample index parameter from the command line.")
    parser.add_argument("--samp_index", type=int, choices=range(1, 6), required=True, help="Sample index must be one of {1, 2, 3, 4, 5}")
    args = parser.parse_args()

    reward_model = RewardModel(ds=6, da=3)

    loaded_files = reward_model.load(load_step=25000)
    if not loaded_files:
        print("No pre-trained model found. A new model will be trained.")

    is_data = reward_model.add_data()
    if is_data:
        reward_model.uniform_sample(samp_index=args.samp_index)