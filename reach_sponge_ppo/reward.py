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

import scipy.stats as st  # ✅
from scipy.stats import norm # ✅

from prompt import (
    traj_pair_process,  stringify_trajs
) # ✅

from logger import logger

# logger for get_queries method
queries_logger = logger.queries_logger
# logger for add_data method
data_logger = logger.data_logger
# logger for training method
train_logger = logger.train_logger
# logger for put queries method
put_logger = logger.put_logger
# logger for get true labels method
true_labels_logger = logger.true_labels_logger
# logger for get pred labels method
pred_labels_logger = logger.pred_labels_logger
# logger for get llm labels method
llm_labels_logger = logger.llm_labels_logger
# logger for save model method 
save_model_logger = logger.save_model_logger
# logger for load model method
load_model_logger = logger.load_model_logger


device = "cuda" if torch.cuda.is_available() else "cpu" # use cuda if available  # ✅


# -----------------------1. Utility Functions -----------------------
import torch.nn as nn

def gen_net(in_size=1, out_size=1, H=128, n_layers=3, activation='tanh', use_batch_norm=True):  # ✅
    """
    Constructs a neural network with optional batch normalisation.

    Args:
        in_size (int): The size of the input layer.
        out_size (int): The size of the output layer.
        H (int): The number of neurons in each hidden layer.
        n_layers (int): The number of hidden layers in the network.
        activation (str): The activation function for the output layer.
        use_batch_norm (bool): Whether to use batch normalisation after each linear layer.

    Returns:
        list: A list of PyTorch layers representing the neural network.
    """

    net = []
    for i in range(n_layers):
        net.append(nn.Linear(in_size, H))
        if use_batch_norm:
            net.append(nn.BatchNorm1d(H))  
        net.append(nn.LeakyReLU())  
        in_size = H
    
    net.append(nn.Linear(in_size, out_size))
    
    # Output activation function
    if activation == 'tanh':
        net.append(nn.Tanh())
    elif activation == 'sig':
        net.append(nn.Sigmoid())
    else:
        net.append(nn.ReLU())

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
            mb_size = 20,      # mini-batch size used during training. Change to 50 for RCW ✅
            activation='tanh',  # activation function to use in the outer layer of the networks ✅
            capacity=1e5,       # total capacity of the buffer that stores training transitions ✅
            size_segment=1,     # length of each trajectory segment ✅
            # env_maker=None,     # optional function or configuration to create the environment ✅
            max_size=10,       # maximum number of training trajectories to keep (for FIFO logic) ✅  # Change to 10 to simulate (10x50) = 500 timesteps.
            large_batch=1,      # multiplier to increase the effective batch size during query sampling ✅
            label_margin=0.0,   # margin used when constructing soft target labels ✅
            teacher_beta=1,    # parameter for a Bradley-Terry model; if > 0, used to compute soft labels ✅
            teacher_gamma=1,    # decay factor applied to earlier parts of a trajectory when computing “rational” labels ✅
            teacher_eps_mistake=0,  # probability of intentionally flipping a label (simulating teacher mistakes) ✅
            teacher_eps_skip=0,     # threshold modifier to skip queries with very high rewards ✅
            teacher_eps_equal=0,    # threshold modifier to mark two segments as equally preferable. ✅
            env_name=None,      # the name of the environment, used for formatting prompts or saving data. ✅
            traj_action=False,   # if True, each input will be a concatenation of state and action (trajectory action). ✅
            traj_save_path=None,# file path to save trajectories (for later inspection or debugging). ✅
            vlm_label=None,     # specifies if a visual language model (or similar) is used for labeling. ✅
            better_traj_gen=False,  # flag indicating whether to try to generate an improved trajectory. ✅
            double_check=False,     # if True, perform a double-check (swapped order) of labeling. ✅
            save_equal=True,    # if True, store queries with equal preference labels as given by the teacher. ✅
            vlm_feedback=True,  # flag indicating whether to use feedback from a visual language model. ✅
            generate_check=False,   # flag for additional checks during generation. ✅
            env_id = None,  # Stores environment ID to track multi-processing 


        ):

        # train data is trajectories, must process to sa and s..   
        self.env_name = env_name # ✅
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
        self.original_size_segment = size_segment # ✅
        self.vlm_label = vlm_label # ✅
        self.better_traj_gen = better_traj_gen # ✅
        self.double_check = double_check # ✅
        self.save_equal = save_equal # ✅
        self.vlm_feedback = vlm_feedback # ✅
        self.generate_check = generate_check # ✅
        self.capacity = int(capacity) # ✅
        
        # Create the buffer for storing the training data. The buffer consists of four parts: buffer_seg1, buffer_seg2, buffer_label and llm_labels_buffer. 
        # buffer_seg1 and buffer_seg2 store the two segments of the trajectory, each of which is a sequence of states and actions. 
        # buffer_label stores the labels assigned to each segment by the teacher. 
        # llm_labels_buffer stores the ground truth labels, which are the true labels assigned by the teacher.
        # The buffer is a numpy array of shape (capacity, size_segment, ds or ds+da), where capacity is the maximum number of trajectories that can be stored, 
        # size_segment is the length of each trajectory segment, and ds+da is the size of the state+action vector if used.
        # If traj_action is True, the buffer stores the full state-action vectors, otherwise it only stores the state vectors.
        if traj_action:
            self.buffer_seg1 = np.empty((self.capacity, size_segment, self.ds+self.da), dtype=np.float32) # ✅
            self.buffer_seg2 = np.empty((self.capacity, size_segment, self.ds+self.da), dtype=np.float32) # ✅
        else:
            self.buffer_seg1 = np.empty((self.capacity, size_segment, self.ds), dtype=np.float32) # ✅
            self.buffer_seg2 = np.empty((self.capacity, size_segment, self.ds), dtype=np.float32) # ✅
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

        self.construct_ensemble() # ✅called to create the ensemble of reward predictors. An Adam optimizer is created over the combined parameters of all networks for optimisation.
        
        self.obses = []  # ✅
        self.pred_rewards = [] # ✅
        self.true_rewards = [] # ✅
        self.actions = [] # ✅
        self.dones = [] # ✅
        self.truncateds = [] # ✅

        self.mb_size = mb_size # ✅
        self.origin_mb_size = mb_size # ✅
        self.train_batch_size = 128  # ✅

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
        self.teacher_thres_skip = 0 # ✅
        self.teacher_thres_equal = 0 # ✅
        
        self.label_margin = label_margin # ❌
        self.label_target = 1 - 2*self.label_margin # ❌

        # for llms
        self.llm_query_accuracy = 0 # ✅
        self.llm_label_accuracy = 0 # ✅

        self.device = device # ✅

    
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
    
   
   
    # --------------------------2. Scheduling and batch size --------------------------

    def change_batch(self, new_frac): # ✅
        """
        Change the batch size of the reward model by a fraction of the original batch size
        
        Parameters
        ----------
        new_frac : float
            The fraction of the original batch size to use
        """
        self.mb_size = int(self.origin_mb_size*new_frac)
    

    def set_batch(self, new_batch): # ✅
        """
        Set the batch size of the reward model. For flexibility.
        
        Parameters
        ----------
        new_batch : int
            The new batch size
        """
        self.mb_size = int(new_batch)


    def set_teacher_thres_skip(self, new_margin): # ✅
        """
        Set the threshold for skipping teacher labels based on the margin to the teacher's skip threshold. This is used to determine when to skip a label in the teacher data based on the margin to the teacher's skip threshold.

        Parameters
        ----------
        new_margin : float
            The new margin to the teacher's skip threshold.

        Returns
        -------
        None
        """
        self.teacher_thres_skip = new_margin * self.teacher_eps_skip
        

    def set_teacher_thres_equal(self, new_margin): # ✅
        """
        Set the threshold for treating teacher labels as equal based on the margin to the teacher's equal threshold. This is used to determine when to treat a label in the teacher data as equal to another label based on the margin to the teacher's equal threshold.

        Parameters
        ----------
        new_margin : float
            The new margin to the teacher's equal threshold.

        Returns
        -------
        None
        """
        self.teacher_thres_equal = new_margin * self.teacher_eps_equal



    # --------------------------3. Trajectory Data Storage --------------------------

    def add_data(self, traj_obs, traj_act, traj_true_rew, traj_pred_rew, traj_dones, traj_truncateds): # ✅
        """
        Add a complete trajectory to the reward model's inputs buffer while maintaining a fixed buffer size.

        Parameters
        ----------
        traj_obs : list or array_like
            A list or 2D array of shape (T, ds) containing the trajectory observations.
        traj_act : list or array_like
            A list or 2D array of shape (T, da) containing the trajectory actions.
        traj_true_rew : list or array_like
            A list or 1D array of length T containing the true rewards.
        traj_pred_rew : list or array_like
            A list or 1D array of length T containing the predicted rewards.
        traj_dones : list or array_like
            A list or 1D array of length T indicating termination flags.
        traj_truncateds : list or array_like
            A list or 1D array of length T indicating truncation flags.

        Returns
        -------
        None
        """
        
        print("Adding trajectory to RM buffers...")
        data_logger.log("Adding trajectory to RM buffers...")
        
        if len(traj_obs) < self.size_segment:
            print("Trajectory length is smaller than the window size.")
            data_logger.log("Trajectory length is smaller than the window size. for querying later. Not added to buffer")
            return
        else:
            # Convert inputs to numpy arrays
            traj_obs = np.array(traj_obs)  # shape (T, ds)
            traj_act = np.array(traj_act)  # shape (T, da)
            traj_true_rew = np.array(traj_true_rew).reshape(-1, 1)  # shape (T, 1)
            traj_pred_rew = np.array(traj_pred_rew).reshape(-1, 1)  # shape (T, 1)
            traj_dones = np.array(traj_dones).reshape(-1, 1)  # shape (T, 1)
            traj_truncateds = np.array(traj_truncateds).reshape(-1, 1)  # shape (T, 1)

            # Combine observations and actions if needed
            if self.traj_action:
                sa_traj = np.concatenate([traj_obs, traj_act], axis=-1)  # shape (T, ds+da)
            else:
                sa_traj = traj_obs  # shape (T, ds)

            # If buffer is full, remove the oldest entry to maintain fixed size
            if len(self.obses) >= self.max_size:
                self.obses.pop(0)
                self.actions.pop(0)
                self.true_rewards.pop(0)
                self.pred_rewards.pop(0)
                self.dones.pop(0)
                self.truncateds.pop(0)

            # Append the new trajectory
            self.obses.append(sa_traj)
            self.actions.append(traj_act)
            self.true_rewards.append(traj_true_rew)
            self.pred_rewards.append(traj_pred_rew)
            self.dones.append(traj_dones)
            self.truncateds.append(traj_truncateds)
        
            data_logger.log("Added trajectory to RM buffers.")
            print("Added trajectory to RM buffers.")
            data_logger.log(f"Current buffer size: {len(self.obses)}")
            data_logger.log(f"Observations: {np.array_str(self.obses)}")
            data_logger.log(f"Actions: {np.array_str(self.actions)}")
            data_logger.log(f"True rewards: {np.array_str(self.true_rewards)}")
            data_logger.log(f"Predicted rewards: {np.array_str(self.pred_rewards)}")
            data_logger.log(f"Dones: {np.array_str(self.dones)}")
            data_logger.log(f"Truncateds: {np.array_str(self.truncateds)}")
            
              
    def add_data_batch(self, obses, actions, true_rews, pred_rews, dones): # ✅❌
        """
        Add a batch of data to the model.
        
        Parameters
        ----------
        obses : array_like
            A batch of observations, where each observation is a 2D array
            of shape `(T, ds)`.
        actions : array_like
            A batch of actions, where each action is a 2D array of shape
            `(T, da)`.
        true_rews : array_like
            A batch of true rewards, where each reward is a 2D array of shape
            `(T,)`.
        pred_rews : array_like
            A batch of predicted rewards, where each reward is a 2D array of shape
            `(T,)`.
        dones : array_like
            A batch of done flags, where each flag is a 2D array of shape
            `(T,)`.
        
        Notes
        -----
        The data is stored in a FIFO buffer with a maximum size of
        `max_size`. If the buffer is full and `done` is `True`, the first
        element of the buffer is discarded and a new element is added to the
        end of the buffer.
        """
        num_env = obses.shape[0]
        for index in range(num_env):
            self.obses.append(obses[index])
            self.actions.append(actions[index])
            self.true_rewards.append(true_rews[index])
            self.pred_rewards.append(pred_rews[index])
       


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
        return r_hats.mean(dim=0)

    
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



    # --------------------------5. Checkpointing --------------------------

    # state_dict() Returns a dictionary containing the model's parameters. Save the model's state dictionary to disc - before rl_model.save()
    def save(self, model_dir, step): # ✅
        """
        Save the state dictionaries of all ensemble members to disc

        Parameters
        ----------
        model_dir : str
            The directory where the model files will be saved.
        step : int
            The current training step or iteration, used for naming the saved model files.

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
            file_path = os.path.join(model_dir, f"reward_model_{step}_{member}.pt")
            torch.save(self.ensemble[member].state_dict(), file_path)
            saved_files.append(file_path)

        print("Reward model saved successfully.")
        save_model_logger.info("Reward model saved successfully.")

        return saved_files  # Return list of saved file paths
    
    

    def load(self, model_dir, step): # ✅
        """
        Load the state dictionaries of all ensemble members from disc.

        Parameters
        ----------
        model_dir : str
            The directory where the model files are saved.
        step : int
            The training step or iteration number used in naming the saved model files.

        Returns
        -------
        list
            A list of file paths of successfully loaded models.
        """
        print("Loading reward model...")
        load_model_logger.info("Loading reward model...")

        loaded_files = []
        for member in range(self.de):
            file_path = os.path.join(model_dir, f"reward_model_{step}_{member}.pt")

            if not os.path.exists(file_path):
                print(f"Warning: Model file {file_path} not found.")
                load_model_logger.warning(f"Model file {file_path} not found.")
                continue  # Skip missing models instead of crashing

            with torch.no_grad():  # Prevent tracking gradients while loading
                self.ensemble[member].load_state_dict(torch.load(file_path))

            loaded_files.append(file_path)

        print("Reward model loaded successfully.")
        load_model_logger.info("Reward model loaded successfully.")

        return loaded_files  # Return list of loaded file paths
    


    # --------------------------6. Getting Labels --------------------------

    # ensemble_acc: Accuracy with respect to the assigned labels.
    # gt_ensemble_acc: Accuracy with respect to the ground-truth labels
    def get_train_acc(self):  # ✅ - NOT USED 
        # ensemble_acc will accumulate the number of correct predictions (matches between the predicted ranking and the assigned labels) for each ensemble member.
        # gt_ensemble_acc will do the same for the ground-truth labels
        # Initialise arrays
        ensemble_acc = np.array([0 for _ in range(self.de)])
        gt_ensemble_acc = np.array([0 for _ in range(self.de)])
        # This value (max_len) is used to know how many stored samples (or trajectory pairs) are available for evaluation.
        max_len = self.capacity if self.buffer_full else self.buffer_index
        # Randomising the order helps ensure that the evaluation is not biased by the order in which data was stored
        total_batch_index = np.random.permutation(max_len)
        # Processing data in batches prevents memory overload and enables vectorized operations for speed
        batch_size = 256
        num_epochs = int(np.ceil(max_len/batch_size))
        
        # This counter keeps track of the total number of samples (or transitions) processed across all batches. It is used later to normalise the accumulated correct predictions into an accuracy value.
        total = 0
        # for each batch
        for epoch in range(num_epochs):
            # Calculate start and end indices for the current batch.
            start_idx = epoch * batch_size
            end_idx = min((epoch + 1) * batch_size, max_len)
            batch_indices = total_batch_index[start_idx:end_idx]
            
            # Slice the stored data using the randomized indices.
            sa_t_1 = self.buffer_seg1[batch_indices]
            sa_t_2 = self.buffer_seg2[batch_indices]
            labels = self.pred_labels_buffer[batch_indices]
            gt_labels = self.llm_labels_buffer[batch_indices]

            # Convert labels to torch tensors and move to the appropriate device.
            labels = torch.from_numpy(labels.flatten()).long().to(device)
            gt_labels = torch.from_numpy(gt_labels.flatten()).long().to(device)

            total += labels.size(0)  # Accumulate the total number of samples processed
            for member in range(self.de):  # for each ensemble member
                # Get the predicted rewards for the two trajectory segments.
                r_hat1 = self.r_hat_member(sa_t_1, member=member)
                r_hat2 = self.r_hat_member(sa_t_2, member=member)
                
                # Sum the rewards along the time (or feature) dimension.
                r_hat1 = r_hat1.sum(axis=1)
                r_hat2 = r_hat2.sum(axis=1)

                # Concatenate the summed rewards to create a two-column tensor.
                # Unsqueezing to ensure proper shape.
                r_hat = torch.cat([r_hat1.unsqueeze(1), r_hat2.unsqueeze(1)], dim=1) 

                # Get the index of the maximum reward (the model's ranking decision).
                _, predicted = torch.max(r_hat, dim=1)
                
                # Compare predictions with the assigned and ground-truth labels.
                correct = (predicted == labels).sum().item()
                gt_correct = (predicted == gt_labels).sum().item()

                # Accumulate the correct counts.
                ensemble_acc[member] += correct
            gt_ensemble_acc[member] += gt_correct

        # These values represent the performance of each ensemble member across the entire training dataset.        
        ensemble_acc = ensemble_acc / total
        gt_ensemble_acc = gt_ensemble_acc / total

        # summary measure of how well the reward model is performing.
        return np.mean(ensemble_acc), np.mean(gt_ensemble_acc)


    # Get queries of given window size (segment) for labelling and preference-assignment
    def get_queries(self, mb_size=20):  # ✅
        print("Getting queries...")
        queries_logger.info("Getting queries...")

        if len(self.obses) == 0:
            queries_logger.error("No trajectory data available.")
            raise ValueError("No trajectory data available.")
        
        # Fixed window size for query segments.
        window_size = self.size_segment 
        
        # Convert lists of trajectories into NumPy arrays with dtype=object.
        train_obses = np.array(self.obses, dtype=object)
        train_actions = np.array(self.actions, dtype=object)
        train_true_rewards = np.array(self.true_rewards, dtype=object)
        train_pred_rewards = np.array(self.pred_rewards, dtype=object)
        train_dones = np.array(self.dones, dtype=object)
        train_truncated = np.array(self.truncateds, dtype=object)
        
        # Randomly sample trajectory indices for both query sets.
        batch_idx_1 = np.random.choice(len(self.obses), size=mb_size, replace=True)
        batch_idx_2 = np.random.choice(len(self.obses), size=mb_size, replace=True)
        
        # Retrieve full trajectories for both query sets.
        obs1_list = train_obses[batch_idx_1]
        acts1_list = train_actions[batch_idx_1]
        true_r1_list = train_true_rewards[batch_idx_1]
        pred_r1_list = train_pred_rewards[batch_idx_1]
        dones1_list = train_dones[batch_idx_1]
        truncated1_list = train_truncated[batch_idx_1]
        
        obs2_list = train_obses[batch_idx_2]
        acts2_list = train_actions[batch_idx_2]
        true_r2_list = train_true_rewards[batch_idx_2]
        pred_r2_list = train_pred_rewards[batch_idx_2]
        dones2_list = train_dones[batch_idx_2]
        truncated2_list = train_truncated[batch_idx_2]
        
        # Helper function that samples a segment of fixed window_size from a list of trajectories, ensuring that exactly mb_size segments are returned (if possible).
        def sample_segments(traj_list, window_size, mb_size):
            """
            Sample mb_size segments of length window_size from traj_list.
            Trajectories that do not meet the window size are skipped.
            If there are not enough trajectories, a ValueError is raised.
            """
            available_trajs = traj_list.copy()
            segments = []
            while len(segments) < mb_size:
                if not available_trajs:
                    print("Not enough trajectories to sample the required number of segments.")
                    return np.array([])
                # Randomly select one trajectory from the available list.
                traj = np.random.choice(available_trajs)
                traj_length = len(traj)
                if traj_length < window_size:
                    print("Trajectory length is smaller than the fixed window size.")
                    available_trajs.remove(traj)
                    continue
                start = np.random.randint(0, traj_length - window_size + 1)
                segments.append(traj[start:start + window_size])
                available_trajs.remove(traj)
            return np.array(segments)
        
        # Sample segments from each query set.
        obs1 = sample_segments(list(obs1_list), window_size, mb_size)
        acts1 = sample_segments(list(acts1_list), window_size, mb_size)
        true_r1 = sample_segments(list(true_r1_list), window_size, mb_size)
        pred_r1 = sample_segments(list(pred_r1_list), window_size, mb_size)
        dones1 = sample_segments(list(dones1_list), window_size, mb_size)
        truncated1 = sample_segments(list(truncated1_list), window_size, mb_size)
        
        obs2 = sample_segments(list(obs2_list), window_size, mb_size)
        acts2 = sample_segments(list(acts2_list), window_size, mb_size)
        true_r2 = sample_segments(list(true_r2_list), window_size, mb_size)
        pred_r2 = sample_segments(list(pred_r2_list), window_size, mb_size)
        dones2 = sample_segments(list(dones2_list), window_size, mb_size)
        truncated2 = sample_segments(list(truncated2_list), window_size, mb_size)

        queries_logger.info("Got queries successfully.")
        print("Got queries successfully.")
        queries_logger.debug(f"obs1: {obs1}")
        queries_logger.debug(f"acts1: {acts1}")
        queries_logger.debug(f"true_r1: {true_r1}")
        queries_logger.debug(f"pred_r1: {pred_r1}")
        queries_logger.debug(f"dones1: {dones1}")
        queries_logger.debug(f"truncated1: {truncated1}")
        queries_logger.debug(f"obs2: {obs2}")
        queries_logger.debug(f"acts2: {acts2}")
        queries_logger.debug(f"true_r2: {true_r2}")
        queries_logger.debug(f"pred_r2: {pred_r2}")
        queries_logger.debug(f"dones2: {dones2}")
        queries_logger.debug(f"truncated2: {truncated2}")
        
        return obs1, obs2, acts1, acts2, true_r1, true_r2, pred_r1, pred_r2, dones1, dones2, truncated1, truncated2


    # This method stores the sliced out (size == self.size_segment) query pairs (and their labels) into circular buffers. It handles buffer overflow with a FIFO (First-In-First-Out) mechanism.
    def put_queries(self, sa_t_1, sa_t_2, pred_labels, true_labels, llm_labels, fake_traj=False): # ✅

        print("Storing queries...")
        put_logger.info("Storing queries...")

        # Validate input shapes: ensure all arrays have the same number of samples.
        num_samples = sa_t_1.shape[0]
        assert sa_t_1.shape[0] == sa_t_2.shape[0] == pred_labels.shape[0] == llm_labels.shape[0] == true_labels.shape[0], "All inputs must have the same number of samples"

        # Determine where to insert the new queries in the buffer.
        next_index = self.buffer_index + num_samples

        if next_index >= self.capacity:
            # Buffer will wrap around.
            self.buffer_full = True

            # Calculate remaining space in the buffer.
            remaining_space = self.capacity - self.buffer_index

            # Insert data into the remaining slots.
            np.copyto(self.buffer_seg1[self.buffer_index:], sa_t_1[:remaining_space])
            np.copyto(self.buffer_seg2[self.buffer_index:], sa_t_2[:remaining_space])
            np.copyto(self.pred_labels_buffer[self.buffer_index:], pred_labels[:remaining_space])
            np.copyto(self.llm_labels_buffer[self.buffer_index:], llm_labels[:remaining_space])
            np.copyto(self.true_labels_buffer[self.buffer_index:], true_labels[:remaining_space])
            self.fake_flag[self.buffer_index:] = fake_traj if np.isscalar(fake_traj) else fake_traj[:remaining_space]

            # Wrap around: Insert remaining data at the beginning of the buffer.
            remaining_samples = num_samples - remaining_space
            if remaining_samples > 0:
                np.copyto(self.buffer_seg1[:remaining_samples], sa_t_1[remaining_space:])
                np.copyto(self.buffer_seg2[:remaining_samples], sa_t_2[remaining_space:])
                np.copyto(self.pred_labels_buffer[:remaining_samples], pred_labels[remaining_space:])
                np.copyto(self.llm_labels_buffer[:remaining_samples], llm_labels[remaining_space:])
                np.copyto(self.true_labels_buffer[:remaining_samples], true_labels[remaining_space:])
                self.fake_flag[:remaining_samples] = fake_traj if np.isscalar(fake_traj) else fake_traj[remaining_space:]

            # Update the buffer index to the new starting position.
            self.buffer_index = remaining_samples

        else:
            # There is enough space in the buffer; insert directly.
            np.copyto(self.buffer_seg1[self.buffer_index:next_index], sa_t_1)
            np.copyto(self.buffer_seg2[self.buffer_index:next_index], sa_t_2)
            np.copyto(self.pred_labels_buffer[self.buffer_index:next_index], pred_labels)
            np.copyto(self.llm_labels_buffer[self.buffer_index:next_index], llm_labels)
            np.copyto(self.true_labels_buffer[self.buffer_index:next_index], true_labels)
            self.fake_flag[self.buffer_index:next_index] = fake_traj if np.isscalar(fake_traj) else fake_traj

            # Move the buffer index forward.
            self.buffer_index = next_index

        print("Stored queries successfully...")
        put_logger.info("Stored queries successfully...")
        put_logger.debug(f"Buffer Segment 1: {np.array_str(self.buffer_seg1)}")
        put_logger.debug(f"Buffer Segment 2: {np.array_str(self.buffer_seg2)}")
        put_logger.debug(f"Buffer True Labels: {np.array_str(self.true_labels_buffer)}")
        put_logger.debug(f"Buffer LLM Labels: {np.array_str(self.llm_labels_buffer)}")
        put_logger.debug(f"Buffer pred Labels: {np.array_str(self.pred_labels_buffer)}")


    # assign labels (or preferences) to a pair of trajectory segments based on their predicted rewards   
    def get_label(self, obs1, obs2, acts1, acts2, true_r1, true_r2, pred_r1, pred_r2, dones1, dones2, truncated1, truncated2): # ✅
        """
        Compute labels for pairs of trajectory segments based on their rewards.
        Returns:
        obs1, obs2, true_r1, true_r2, pred_r1, pred_r2, true_labels, pred_labels, llm_labels
        """
        # Compute summed rewards along time dimension.
        sum_true_r1 = np.sum(true_r1, axis=1)
        sum_true_r2 = np.sum(true_r2, axis=1)
        sum_pred_r1 = np.sum(pred_r1, axis=1)
        sum_pred_r2 = np.sum(pred_r2, axis=1)

        # Filter out queries where both segments are not informative enough.
        if self.teacher_thres_skip > 0:
            max_true_r = np.maximum(sum_true_r1, sum_true_r2)
            max_pred_r = np.maximum(sum_pred_r1, sum_pred_r2)
            valid_true_mask = max_true_r > self.teacher_thres_skip
            valid_pred_mask = max_pred_r > self.teacher_thres_skip
            if np.sum(valid_true_mask) == 0 and np.sum(valid_pred_mask) == 0:
                return None, None, None, None, None, None, [], [], []
        else:
            # If teacher_thres_skip is not positive (ie, not being used), all queries are valid.
            valid_true_mask = np.ones_like(sum_true_r1, dtype=bool)
            valid_pred_mask = np.ones_like(sum_pred_r1, dtype=bool)

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

        # Call helper methods to compute labels.
        true_labels = self.get_true_labels(obs1_true, obs2_true, acts1_true, acts2_true, true_r1_filt, true_r2_filt)
        pred_labels = self.get_pred_labels(obs1_pred, obs2_pred, acts1_pred, acts2_pred, pred_r1_filt, pred_r2_filt)
        llm_labels = self.get_llm_labels(obs1, obs2, acts1_pred, acts2_pred, pred_r1_filt, pred_r2_filt, pred_labels, true_labels, dones1_pred, dones2_pred, truncated1_pred, truncated2_pred)   

        return obs1, obs2, true_r1, true_r2, pred_r1, pred_r2, true_labels, pred_labels, llm_labels


    def get_true_labels(self, obs1, obs2, acts1, acts2, r1, r2): # ✅
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
        sum_r1 = np.sum(temp_r1, axis=1)
        sum_r2 = np.sum(temp_r2, axis=1)

        # Identify ambiguous queries where the absolute difference is below a threshold.
        margin_index = np.abs(sum_r1 - sum_r2) < self.teacher_thres_equal

        # Rational labels: label 1 if first segment reward is less than second, else 0.
        rational_labels = (sum_r1 < sum_r2).astype(int)

        # Optionally refine labels using a Bradley-Terry style model.
        if self.teacher_beta > 0:
            # Convert to torch tensors.
            r_hat = torch.cat([
                torch.tensor(sum_r1, dtype=torch.float32).unsqueeze(1),
                torch.tensor(sum_r2, dtype=torch.float32).unsqueeze(1)
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
        return gt_labels


    def get_pred_labels(self, obs1, obs2, acts1, acts2, r1, r2): # ✅
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

        for index in range(seg_size - 1):
            temp_r1[:, :index + 1] *= self.teacher_gamma
            temp_r2[:, :index + 1] *= self.teacher_gamma

        sum_r1 = np.sum(temp_r1, axis=1)
        sum_r2 = np.sum(temp_r2, axis=1)

        margin_index = np.abs(sum_r1 - sum_r2) < self.teacher_thres_equal

        rational_labels = (sum_r1 < sum_r2).astype(int)

        if self.teacher_beta > 0:
            r_hat = torch.cat([
                torch.tensor(sum_r1, dtype=torch.float32).unsqueeze(1),
                torch.tensor(sum_r2, dtype=torch.float32).unsqueeze(1)
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
        return pred_labels


    def get_llm_labels(self, obs1, obs2, acts1, acts2, r1, r2, pred_labels, true_labels, dones1, dones2, truncated1, truncated2): # ✅
        """
        Compute LLM labels for pairs of trajectory segments using predicted rewards.
        
        Each query corresponds to a trajectory segment (of length self.size_segment). For each segment pair,an LLM (via GPT inference) is queried to decide which segment is preferable. Actions are included in the prompt for additional context.
        
        Additionally, two LLM label accuracies are computed:
        - One with respect to the predicted rewards labels.
        - One with respect to the true rewards labels.
        
        Parameters:
        obs1, obs2 : array-like
            The observation trajectories for the two segments.
        acts1, acts2 : array-like
            The action trajectories for the two segments.
        r1, r2 : array-like
            The reward trajectories for the two segments.
        pred_labels : numpy array
            The labels derived from predicted rewards.
        true_labels : numpy array
            The labels derived from true rewards.
        
        Returns:
        llm_labels : numpy array
            An array of inferred labels (0 or 1) for each query, or -1 if ambiguous/invalid.
        """
        
        print("Computing LLM labels...")
        llm_labels_logger.info("Computing LLM labels...")
        
        # Save the current buffer index.
        origin_index = self.buffer_index

        # Initialise an array for LLM labels, defaulting to -1 (invalid/ambiguous prediction).
        llm_labels = np.full(pred_labels.shape, -1, dtype=int)

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
                truncated2[idx]
            )
            
            # Format the trajectory pair into text strings for the GPT prompt.
            traj_str1, traj_str2 = stringify_trajs(traj_1, traj_2)

            # Call the GPT inference using the constructed prompt.
            answer = gpt_infer(traj_str1, traj_str2)

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

            # If double-checking is enabled and the first answer is valid, perform a swapped query.
            if self.double_check and label_res != -1:
                # Swap the order of the trajectories.
                answer_swapped = gpt_infer(traj_str2, traj_str1)
                try:
                    label_res_swapped = int(answer_swapped)
                    if label_res_swapped in [1, 2]:
                        label_res_swapped -= 1
                    else:
                        label_res_swapped = -1
                except Exception:
                    label_res_swapped = -1

                # If the two answers are not opposite, mark the label as invalid.
                if not ((label_res == 0 and label_res_swapped == 1) or (label_res == 1 and label_res_swapped == 0)):
                    print("Double check False!")
                    label_res = -1

            # Store the inferred label for this query.
            llm_labels[idx] = label_res

            # Optionally save the processed data and the raw LLM response.
            if self.traj_save_path is not None:
                os.makedirs(self.traj_save_path, exist_ok=True)
                # Create a dictionary to save.
                data_to_save = {
                    "traj1": traj_1,
                    "traj2": traj_2,
                    "llm_response": answer,
                    "llm_response_swapped": answer_swapped,
                    "label": label_res,
                    "label_swapped": label_res_swapped
                }
                data_file_path = os.path.join(self.traj_save_path, f"traj_pairs_{origin_index+idx}.json")
                with open(data_file_path, 'w') as data_file:
                    json.dump(data_to_save, data_file, indent=4, default=lambda o: o.__dict__)

        # After processing, compute two accuracy measures on valid predictions.
        valid_indices = [i for i, lab in enumerate(llm_labels) if lab != -1]
        if valid_indices:
            filtered_llm = llm_labels[valid_indices]
            filtered_pred_labels = pred_labels[valid_indices]
            filtered_true_labels = true_labels[valid_indices]
            correct_pred = sum(1 for vlm_lab, p in zip(filtered_llm, filtered_pred_labels) if vlm_lab == p)
            accuracy_pred = correct_pred / len(filtered_llm)
            correct_true = sum(1 for vlm_lab, t in zip(filtered_llm, filtered_true_labels) if vlm_lab == t)
            accuracy_true = correct_true / len(filtered_llm)
            print(f"LLM Label Accuracy (Predicted): {accuracy_pred * 100:.2f}%")
            print(f"LLM Label Accuracy (True): {accuracy_true * 100:.2f}%")
            llm_labels_logger.info(f"LLM Label Accuracy (Predicted): {accuracy_pred * 100:.2f}%")
            llm_labels_logger.info(f"LLM Label Accuracy (True): {accuracy_true * 100:.2f}%")
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

    def train_reward(self): # ✅
        """
        Train the reward model using hard targets (cross-entropy loss).
        """

        print("Training reward model...")
        train_logger.info("Training reward model...")

        # Create lists for each ensemble member to store losses.
        ensemble_losses = [[] for _ in range(self.de)]
        ensemble_acc = np.zeros(self.de)
        gt_ensemble_acc = np.zeros(self.de)
        
        # Determine the number of training examples.
        num_examples = self.capacity if self.buffer_full else self.buffer_index
        
        # Create a random permutation of indices for each ensemble member.
        total_batch_index = [np.random.permutation(num_examples) for _ in range(self.de)]
        
        # Compute the number of epochs required.
        num_epochs = int(np.ceil(num_examples / self.train_batch_size))
        
        total_samples = 0  # Count of overall examples used.
        gt_total = 0       # Count of valid (non-fake) examples.
        
        for epoch in range(num_epochs):
            self.opt.zero_grad()
            batch_loss = 0.0
            
            # Determine the batch slice indices.
            last_index = min((epoch + 1) * self.train_batch_size, num_examples)
            
            for member in range(self.de):
                # Get indices for the current mini-batch.
                idxs = total_batch_index[member][epoch * self.train_batch_size:last_index]
                sa_t_1 = self.buffer_seg1[idxs]
                sa_t_2 = self.buffer_seg2[idxs]
                labels = self.pred_labels_buffer[idxs] # predicted labels
                llm_labels = self.llm_labels_buffer[idxs] # llm labels
                fake_labels = self.fake_flag[idxs]
                
                # Convert labels to PyTorch tensors.
                labels_t = torch.from_numpy(labels.flatten()).long().to(device)
                llm_labels_t = torch.from_numpy(llm_labels.flatten()).long().to(device)
                
                # For member 0, update the counters.
                if member == 0:
                    total_samples += labels_t.size(0)
                    # Note: Assuming fake_flag is a boolean array.
                    gt_total += np.sum(fake_labels.flatten() == False)
                
                # Compute logits for each segment.
                r_hat1 = self.r_hat_member(sa_t_1, member=member).sum(dim=1)
                r_hat2 = self.r_hat_member(sa_t_2, member=member).sum(dim=1)
                logits = torch.cat([r_hat1.unsqueeze(1), r_hat2.unsqueeze(1)], dim=-1)
                
                # Compute cross-entropy loss.
                curr_loss = self.CEloss(logits, llm_labels_t)
                batch_loss += curr_loss
                ensemble_losses[member].append(curr_loss.item())
                
                # Get predictions and update accuracy.
                _, predicted = torch.max(logits.data, 1)
                ensemble_acc[member] += (predicted == llm_labels_t).sum().item()
                # Only count accuracy on valid (non-fake) samples.
                valid_indices = np.where(fake_labels.flatten() == False)[0]
                if len(valid_indices) > 0:
                    gt_correct = (predicted[valid_indices] == llm_labels_t[valid_indices]).sum().item()
                    gt_ensemble_acc[member] += gt_correct
            
            batch_loss.backward()
            self.opt.step()
        
        # Compute ensemble accuracies.
        ensemble_acc = ensemble_acc / total_samples
        self.llm_label_accuracy = gt_ensemble_acc / gt_total if gt_total != 0 else 0

        print("Training reward model successfully...")
        print("LLM Label Accuracy: ", self.llm_label_accuracy)
        print("LLM Query Accuracy: ", self.llm_query_accuracy)
        train_logger.info(f"LLM Label Accuracy: {self.llm_label_accuracy}")
        train_logger.info(f"LLM Query Accuracy: {self.llm_query_accuracy}")
        train_logger.info(f"Ensemble Accuracies: {ensemble_acc}")
        train_logger.info(f"Training reward model successfully...")

        
        return self.llm_query_accuracy, self.llm_label_accuracy, ensemble_acc


    def train_soft_reward(self): # ✅ - NOT USED
        """
        Train the reward model using soft targets (soft cross-entropy loss).
        """
        ensemble_losses = [[] for _ in range(self.de)]
        ensemble_acc = np.zeros(self.de)
        gt_ensemble_acc = np.zeros(self.de)
        
        num_examples = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = [np.random.permutation(num_examples) for _ in range(self.de)]
        num_epochs = int(np.ceil(num_examples / self.train_batch_size))
        total_samples = 0
        
        for epoch in range(num_epochs):
            self.opt.zero_grad()
            batch_loss = 0.0
            
            last_index = min((epoch + 1) * self.train_batch_size, num_examples)
            
            for member in range(self.de):
                idxs = total_batch_index[member][epoch * self.train_batch_size:last_index]
                sa_t_1 = self.buffer_seg1[idxs]
                sa_t_2 = self.buffer_seg2[idxs]
                labels = self.pred_labels_buffer[idxs]
                gt_labels = self.llm_labels_buffer[idxs]
                
                labels_t = torch.from_numpy(labels.flatten()).long().to(device)
                gt_labels_t = torch.from_numpy(gt_labels.flatten()).long().to(device)
                
                if member == 0:
                    total_samples += labels_t.size(0)
                
                r_hat1 = self.r_hat_member(sa_t_1, member=member).sum(dim=1)
                r_hat2 = self.r_hat_member(sa_t_2, member=member).sum(dim=1)
                logits = torch.cat([r_hat1.unsqueeze(1), r_hat2.unsqueeze(1)], dim=-1)
                
                # Construct soft targets.
                # For ambiguous cases (label == -1), temporarily set them to 0.
                ambiguous_mask = (labels_t == -1)
                labels_t_mod = labels_t.clone()
                labels_t_mod[ambiguous_mask] = 0
                # Create one-hot targets scaled by self.label_target.
                target_onehot = torch.zeros_like(logits).scatter(1, labels_t_mod.unsqueeze(1), self.label_target)
                # Add a margin for smoothing.
                target_onehot += self.label_margin
                # Force ambiguous cases to have equal probabilities.
                if ambiguous_mask.sum() > 0:
                    target_onehot[ambiguous_mask] = 0.5
                
                curr_loss = self.softXEnt_loss(logits, target_onehot)
                batch_loss += curr_loss
                ensemble_losses[member].append(curr_loss.item())
                
                _, predicted = torch.max(logits.data, 1)
                ensemble_acc[member] += (predicted == labels_t).sum().item()
                gt_ensemble_acc[member] += (predicted == gt_labels_t).sum().item()
            
            batch_loss.backward()
            self.opt.step()
        
        ensemble_acc = ensemble_acc / total_samples
        self.llm_label_accuracy = gt_ensemble_acc / total_samples
        
        return self.llm_query_accuracy, self.llm_label_accuracy, ensemble_acc
    

    # --------------------------8. Sampling -----------------------------------

    # When implemented fully, its goal would be to return a number of selected queries (or their labels) that have been chosen solely based on diversity using the k-center approach
    def kcenter_sampling(self): # PASS
        pass
    

    # The k-center-based methods (including the variants that incorporate disagreement or entropy) aim to choose a diverse subset of queries. This helps ensure that the training data covers different regions of the input space.
    def kcenter_disagree_sampling(self): # PASS
        pass
    

    # This method is very similar to kcenter_disagree_sampling, except that it uses entropy (a measure of uncertainty) instead of disagreement from the ensemble.
    def kcenter_entropy_sampling(self): # PASS
        pass


    # Instead of using uncertainty or diversity measures, it simply randomly samples queries with a batch size equal to mb_size. - Randomly selects queries without any additional criteria. This is the simplest approach and can serve as a baseline.
    def uniform_sampling(self):  # ✅
        # Get queries from the buffer.
        observations1, observations2, actions1, actions2, true_rewards1, true_rewards2, predicted_rewards1, predicted_rewards2, dones1, dones2, truncated1, truncated2 = self.get_queries(mb_size=self.mb_size)
        
        # Compute labels for each query pair.
        # get_label returns:
        #   obs1, obs2,
        #   true_r1, true_r2,
        #   predicted_r1, predicted_r2,
        #   true_labels, predicted_labels, llm_labels
        obs1, obs2, true_r1, true_r2, predicted_r1, predicted_r2, true_labels, predicted_labels, llm_labels = self.get_label(observations1, observations2, actions1, actions2, true_rewards1, true_rewards2, predicted_rewards1, predicted_rewards2, dones1, dones2, truncated1, truncated2)
        
        # *********************************************************************
        # Filter out ambiguous queries.
        # If we do not want to save "equal" (ambiguous) queries (labels == -1), then filter them out.
        if not self.save_equal:
            valid_indices = predicted_labels[:, 0] != -1  # Keep queries with valid (non-ambiguous) predicted labels.
            obs1 = obs1[valid_indices]
            obs2 = obs2[valid_indices]
            true_r1 = true_r1[valid_indices]
            true_r2 = true_r2[valid_indices]
            predicted_r1 = predicted_r1[valid_indices]
            predicted_r2 = predicted_r2[valid_indices]
            true_labels = true_labels[valid_indices]
            predicted_labels = predicted_labels[valid_indices]
            llm_labels = llm_labels[valid_indices]
        
        # If there are any queries left after filtering, store them.
        if len(predicted_labels) > 0:
            self.put_queries(obs1, obs2, predicted_labels, true_labels, llm_labels)
        
        # Return the number of queries (or labels) processed.
        return len(predicted_labels)
   

    # These methods select queries where the ensemble is uncertain (i.e. high disagreement or high entropy). They are useful if you want the model to focus on samples where it is unsure, which can be very informative for training.
    def disagreement_sampling(self): # PASS
        pass
    

    def entropy_sampling(self): # PASS
        pass
