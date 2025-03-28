# sources: 
# RL-SaLLM-F paper: https://github.com/TU2021/RL-SaLLM-F/blob/main/reward_model.py
# Docstrings are automatically generated using codeium but verified manually for correctness

import numpy as np   # ✅
import torch # type: ignore  # ✅
import torch.nn as nn # type: ignore  # ✅
import torch.nn.functional as F # type: ignore  # ✅
import torch.utils.data as data # type: ignore  # ✅
import torch.optim as optim # type: ignore  # ✅
import itertools  # ✅
import tqdm # type: ignore  # ✅
import copy  # ✅
import scipy.stats as st # type: ignore  # ✅
import os  # ✅
import time  # ✅
import json  # ✅

from scipy.stats import norm # type: ignore  # ✅

# from obs_process import traj_pair_process, format_traj_for_gpt, get_response_answer, extract_trajectory_from_text, convert_trajdist_to_traj, format_traj_begin_for_gpt
from prompt import (
    gpt_query_env_prompts,  # ✅
    trajgen_template,  # Ignore
) # Change this to import your gpt prompt instead.

device = 'cuda'  # if you have cuda available  # ✅


def gen_net(in_size=1, out_size=1, H=128, n_layers=3, activation='tanh'): # ✅
    """
    A utility function that constructs a neural network architecture as a list of layers.

    Args:
        in_size (int): The size of the input layer.
        out_size (int): The size of the output layer.
        H (int): The number of neurons in each hidden layer.
        n_layers (int): The number of hidden layers in the network.
        activation (str): The activation function to use for the output layer.
            Options are 'tanh', 'sig', or others defaulting to ReLU.

    Returns:
        list: A list of PyTorch layers representing the neural network to be wrapped in an nn.Sequential module later.
    """

    net = []
    for i in range(n_layers):
        net.append(nn.Linear(in_size, H))
        net.append(nn.LeakyReLU())
        in_size = H
    net.append(nn.Linear(in_size, out_size))
    if activation == 'tanh':
        net.append(nn.Tanh())
    elif activation == 'sig':
        net.append(nn.Sigmoid())
    else:
        net.append(nn.ReLU())

    return net
    
    
def KCenterGreedy(obs, full_obs, num_new_sample): # ✅
    # obs -> a list of candidate observations we want to sample from
    # full_obs -> a list of all observations
    # num_new_sample -> number of new samples (candidates) we want to sample
    
    selected_index = []  # This list will store the indices of the selected observations
    current_index = list(range(obs.shape[0])) # This list is initialised to store the indices of candidate observations in obs
    new_obs = obs  # Initially set to obs. As we select new samples, we update it to only contain the remaining candidate observations.
    new_full_obs = full_obs # Initially set to full_obs. As we select new samples, we update it to only. ***********************
    
    for count in range(num_new_sample):  # Loop until we have selected num_new_sample observations
        dist = compute_smallest_dist(new_obs, new_full_obs)  # calculates, for each candidate in new_obs, the smallest Euclidean distance to any point in new_full_obs.
        max_index = torch.argmax(dist)  # finds the index of the candidate in new_obs with the maximum of these minimum distances. This candidate is the one most “isolated” from the current reference set.
        max_index = max_index.item() # Convert the PyTorch tensor to a Python integer
        
        if count == 0:
            selected_index.append(max_index) # On the very first iteration, since current_index is just a list from 0 to obs.shape[0]-1, appending max_index is sufficient.
        else:
            selected_index.append(current_index[max_index]) # For later iterations, the candidate set new_obs is a subset of the original obs. Therefore, current_index[max_index] maps the index in new_obs back to the corresponding index in the original array.
            
        current_index = current_index[0:max_index] + current_index[max_index+1:]
        new_obs = obs[current_index]
        new_full_obs = np.concatenate([
            full_obs, 
            obs[selected_index]], 
            axis=0)
        
    return selected_index


def compute_smallest_dist(obs, full_obs):  # ✅
    """
    Computes the smallest Euclidean distance between each observation in `obs` and the set of observations in `full_obs`.

    Parameters
    ----------
    obs : ndarray
        An array of observations for which the smallest distances are to be computed.
    full_obs : ndarray
        An array of reference observations to compute distances against.

    Returns
    -------
        A tensor containing the smallest distance for each observation in `obs` to any observation in `full_obs`,
        with shape (len(obs), 1).
    """

    obs = torch.from_numpy(obs).float()
    full_obs = torch.from_numpy(full_obs).float()
    batch_size = 100
    with torch.no_grad():
        total_dists = []
        for full_idx in range(len(obs) // batch_size + 1):
            full_start = full_idx * batch_size
            if full_start < len(obs):
                full_end = (full_idx + 1) * batch_size
                dists = []
                for idx in range(len(full_obs) // batch_size + 1):
                    start = idx * batch_size
                    if start < len(full_obs):
                        end = (idx + 1) * batch_size
                        dist = torch.norm(
                            obs[full_start:full_end, None, :].to(device) - full_obs[None, start:end, :].to(device), dim=-1, p=2
                        )
                        dists.append(dist)
                dists = torch.cat(dists, dim=1)
                small_dists = torch.torch.min(dists, dim=1).values
                total_dists.append(small_dists)
                
        total_dists = torch.cat(total_dists)
    return total_dists.unsqueeze(1)


class RewardModel:
    def __init__(self, #✅
            ds,                 # state dimension: size of the state/observation vector. ✅
            da,                 # action dimension: size of the action vector. ✅
            ensemble_size=3,    # number of reward predictors in the ensemble (i.e., number of separate networks). ✅
            lr=3e-4,            # learning rate for the optimiser. ✅
            mb_size = 128,      # mini-batch size used during training. ✅
            activation='tanh',  # activation function to use in the networks (e.g., tanh) - output layer. ✅
            capacity=5e5,       # total capacity of the buffer that stores training transitions. ✅
            size_segment=1,     # length of each trajectory segment (number of consecutive transitions). ✅
            env_maker=None,     # optional function or configuration to create the environment. ✅
            max_size=100,       # maximum number of training trajectories to keep (for FIFO logic).  *****
            large_batch=1,      # multiplier to increase the effective batch size during query sampling.  ****
            label_margin=0.0,   # margin used when constructing soft target labels. ****
            teacher_beta=-1,    # parameter for a Bradley-Terry model; if >0, used to compute soft labels. ****
            teacher_gamma=1,    # decay factor applied to earlier parts of a trajectory when computing “rational” labels. ****
            teacher_eps_mistake=0,  # probability of intentionally flipping a label (simulating teacher mistakes). ****
            teacher_eps_skip=0,     # threshold modifier to skip queries with very high rewards. ****
            teacher_eps_equal=0,    # threshold modifier to mark two segments as equally preferable. ****
            env_name=None,      # the name of the environment, used for formatting prompts or saving data. ✅
            traj_action=True,   # if True, each input will be a concatenation of state and action (trajectory action). ✅
            traj_save_path=None,# file path to save trajectories (for later inspection or debugging). ✅
            vlm_label=None,     # specifies if a visual language model (or similar) is used for labeling. ✅
            better_traj_gen=False,  # flag indicating whether to try to generate an improved trajectory. ✅
            double_check=False,     # if True, perform a double-check (swapped order) of labeling. ✅
            save_equal=True,    # if True, store queries with equal preference labels as given by the teacher. ✅
            vlm_feedback=True,  # flag indicating whether to use feedback from a visual language model. ✅
            generate_check=False,   # flag for additional checks during generation. ✅
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
        self.max_size = max_size # ****
        self.activation = activation # ✅
        self.size_segment = size_segment # ****
        self.vlm_label = vlm_label # ✅
        self.better_traj_gen = better_traj_gen # ✅
        self.double_check = double_check # ✅
        self.save_equal = save_equal # ✅
        self.vlm_feedback = vlm_feedback # ✅
        self.generate_check = generate_check # ✅
        self.capacity = int(capacity) # ✅
        
        # Create the buffer for storing the training data. The buffer consists of four parts: buffer_seg1, buffer_seg2, buffer_label and gt_buffer_label. 
        # buffer_seg1 and buffer_seg2 store the two segments of the trajectory, each of which is a sequence of states and actions. 
        # buffer_label stores the labels assigned to each segment by the teacher. 
        # gt_buffer_label stores the ground truth labels, which are the true labels assigned by the teacher.
        # The buffer is a numpy array of shape (capacity, size_segment, ds+da), where capacity is the maximum number of trajectories that can be stored, 
        # size_segment is the length of each trajectory segment, and ds+da is the size of the state+action vector.
        # If traj_action is True, the buffer stores the full state-action vectors, otherwise it only stores the state vectors.
        if traj_action:
            self.buffer_seg1 = np.empty((self.capacity, size_segment, self.ds+self.da), dtype=np.float32) # ✅
            self.buffer_seg2 = np.empty((self.capacity, size_segment, self.ds+self.da), dtype=np.float32) # ✅
        else:
            self.buffer_seg1 = np.empty((self.capacity, size_segment, self.ds), dtype=np.float32) # ✅
            self.buffer_seg2 = np.empty((self.capacity, size_segment, self.ds), dtype=np.float32) # ✅
        self.buffer_label = np.empty((self.capacity, 1), dtype=np.float32) # ✅
        self.gt_buffer_label = np.empty((self.capacity, 1), dtype=np.float32) # ✅
        # fake_flag is a boolean array of shape (capacity, 1) that indicates whether a particular entry in the buffer is fake or not.
        # An entry is considered fake if it is generated by the model, rather than being a real trajectory collected from the environment.
        # The buffer_index is the current index in the buffer where the next trajectory segment will be stored.
        # The buffer_full flag is set to True if the buffer is full, i.e., if the buffer_index has reached the capacity of the buffer.
        self.fake_flag = np.empty((self.capacity, 1), dtype=bool) # ✅
        self.buffer_index = 0 # ✅
        self.buffer_full = False # ✅

                
        self.construct_ensemble() # ✅called to create the ensemble of reward predictors. An Adam optimizer is created over the combined parameters of all networks.
        
        self.inputs = []  # ✅
        self.targets = [] # ✅
        self.raw_actions = [] # ****
        self.img_inputs = []  # ✅ - Not required
        self.mb_size = mb_size # ✅
        self.origin_mb_size = mb_size # ✅
        self.train_batch_size = 128  # ✅
        self.CEloss = nn.CrossEntropyLoss() # ✅
        self.running_means = [] # ****
        self.running_stds = [] # ****
        self.best_seg = [] # ****
        self.best_label = [] # ****
        self.best_action = [] # ****
        self.large_batch = large_batch # ****
        
        # new teacher
        self.teacher_beta = teacher_beta # ✅
        self.teacher_gamma = teacher_gamma # ✅
        self.teacher_eps_mistake = teacher_eps_mistake # ****
        self.teacher_eps_equal = teacher_eps_equal # ✅
        self.teacher_eps_skip = teacher_eps_skip # ✅
        self.teacher_thres_skip = 0 # ✅
        self.teacher_thres_equal = 0 # ✅
        
        self.label_margin = label_margin # ****
        self.label_target = 1 - 2*self.label_margin # ****

        # for llms
        self.llm_query_accuracy = 0 # ✅
        self.llm_label_accuracy = 0 # ✅
        
    def construct_ensemble(self): # ✅
        """
        Construct an ensemble of neural network models for reward prediction.

        This function initialises a specified number of neural network models 
        (based on the value of `self.de`) and appends them to the ensemble. 
        Each model is created using a generator function `gen_net` with specific 
        input sizes depending on whether trajectory actions are considered. The 
        models are added to the ensemble list, and their parameters are appended 
        to a parameter list for optimisation. An Adam optimiser is then created 
        over these combined parameters.

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
                                           out_size=1, H=256, n_layers=3, 
                                           activation=self.activation)).float().to(device)
            else:
                model = nn.Sequential(*gen_net(in_size=self.ds, 
                                           out_size=1, H=256, n_layers=3, 
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
        Set the threshold for skipping teacher labels based on the margin to the teacher's
        skip threshold. This is used to determine when to skip a label in the teacher
        data based on the margin to the teacher's skip threshold.

        Parameters
        ----------
        new_margin : float
            The new margin to the teacher's skip threshold.

        Returns
        -------
        None
        """
        self.teacher_thres_skip = new_margin * self.teacher_eps_skip
        
    def set_teacher_thres_equal(self, new_margin):# ✅
        """
        Set the threshold for treating teacher labels as equal based on the margin to the teacher's
        equal threshold. This is used to determine when to treat a label in the teacher data as
        equal to another label based on the margin to the teacher's equal threshold.

        Parameters
        ----------
        new_margin : float
            The new margin to the teacher's equal threshold.

        Returns
        -------
        None
        """
        self.teacher_thres_equal = new_margin * self.teacher_eps_equal
     
     
    def add_data(self, obs, act, rew, done): # ****
        """
        Add data to the reward model's buffer

        Parameters
        ----------
        obs : array_like
            The observation at the current timestep
        act : array_like
            The action taken at the current timestep
        rew : float
            The reward received at the next timestep
        done : bool
            Whether the episode terminated at the next timestep

        Notes
        -----
        The data is stored in a FIFO buffer with a maximum size of `max_size`.
        If the buffer is full and `done` is `True`, the first element of the buffer is discarded
        and a new element is added to the end of the buffer.
        """
        if self.traj_action:
            sa_t = np.concatenate([obs, act], axis=-1)
            flat_input = sa_t.reshape(1, self.da+self.ds)
        else:
            sa_t = obs
            flat_input = sa_t.reshape(1, self.ds)

        r_t = rew
        r_t = np.array(r_t)
        flat_target = r_t.reshape(1, 1)

        init_data = len(self.inputs) == 0
        if init_data:
            self.inputs.append(flat_input)
            self.targets.append(flat_target)
        elif done:
            self.inputs[-1] = np.concatenate([self.inputs[-1], flat_input])
            self.targets[-1] = np.concatenate([self.targets[-1], flat_target])
            # FIFO
            if len(self.inputs) > self.max_size:
                self.inputs = self.inputs[1:]
                self.targets = self.targets[1:]
            self.inputs.append([])
            self.targets.append([])
        else:
            if len(self.inputs[-1]) == 0:
                self.inputs[-1] = flat_input
                self.targets[-1] = flat_target
            else:
                self.inputs[-1] = np.concatenate([self.inputs[-1], flat_input])
                self.targets[-1] = np.concatenate([self.targets[-1], flat_target])   
        
        
    def add_data_batch(self, obses, rewards): # ****
        """
        Add a batch of data to the model.
        
        Parameters
        ----------
        obses : array_like
            A batch of observations, where each observation is a 2D array
            of shape `(T, ds)`.
        rewards : array_like
            A batch of rewards, where each reward is a 2D array of shape
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
            self.inputs.append(obses[index])
            self.targets.append(rewards[index])
       
            
    def get_rank_probability(self, x_1, x_2): # ✅
        # get probability x_1 > x_2
        """
        Compute the probability that x_1 is preferred to x_2.
        
        Parameters
        ----------
        x_1 : array_like
            The first input to compare.
        x_2 : array_like
            The second input to compare.
        
        Returns
        -------
        mean : float
            The mean probability across all ensemble members.
        std : float
            The standard deviation of the probabilities across all ensemble members.
        """
        probs = []
        for member in range(self.de):
            probs.append(self.p_hat_member(x_1, x_2, member=member).cpu().numpy())
        probs = np.array(probs)
        
        return np.mean(probs, axis=0), np.std(probs, axis=0)
    
    
    def get_entropy(self, x_1, x_2): # ****
        # get probability x_1 > x_2
        """
        Calculate the entropy of the probability distribution that x_1 is preferred to x_2.

        Parameters
        ----------
        x_1 : array_like
            The first input to compare.
        x_2 : array_like
            The second input to compare.

        Returns
        -------
        mean_entropy : float
            The mean entropy across all ensemble members.
        std_entropy : float
            The standard deviation of the entropy across all ensemble members.
        """

        probs = []
        for member in range(self.de):
            probs.append(self.p_hat_entropy(x_1, x_2, member=member).cpu().numpy())
        probs = np.array(probs)
        return np.mean(probs, axis=0), np.std(probs, axis=0)
    

    def p_hat_member(self, x_1, x_2, member=-1): # ✅
        # softmaxing to get the probabilities according to eqn 1
        """
        Compute the probability that x_1 is preferred to x_2 according to an ensemble member.
        
        Parameters
        ----------
        x_1 : array_like
            The first input to compare.
        x_2 : array_like
            The second input to compare.
        member : int, optional
            The index of the ensemble member to use. If `-1`, the mean across all ensemble members is used.
        
        Returns
        -------
        probability : float
            The probability that x_1 is preferred to x_2.
        """
        with torch.no_grad():
            r_hat1 = self.r_hat_member(x_1, member=member)
            r_hat2 = self.r_hat_member(x_2, member=member)
            r_hat1 = r_hat1.sum(axis=1)
            r_hat2 = r_hat2.sum(axis=1)
            r_hat = torch.cat([r_hat1, r_hat2], axis=-1)
        
        # taking 0 index for probability x_1 > x_2
        return F.softmax(r_hat, dim=-1)[:,0]
    
    
    def p_hat_entropy(self, x_1, x_2, member=-1): # ****
        # softmaxing to get the probabilities according to eqn 1
        """
        Compute the entropy of the probability distribution that x_1 is preferred to x_2 according to an ensemble member.
        
        Parameters
        ----------
        x_1 : array_like
            The first input to compare.
        x_2 : array_like
            The second input to compare.
        member : int, optional
            The index of the ensemble member to use. If `-1`, the mean across all ensemble members is used.
        
        Returns
        -------
        entropy : float
            The entropy of the probability distribution that x_1 is preferred to x_2.
        """
        with torch.no_grad():
            r_hat1 = self.r_hat_member(x_1, member=member)
            r_hat2 = self.r_hat_member(x_2, member=member)
            r_hat1 = r_hat1.sum(axis=1)
            r_hat2 = r_hat2.sum(axis=1)
            r_hat = torch.cat([r_hat1, r_hat2], axis=-1)
        
        ent = F.softmax(r_hat, dim=-1) * F.log_softmax(r_hat, dim=-1)
        ent = ent.sum(axis=-1).abs()
        return ent


    def r_hat_member(self, x, member=-1): # ✅
        # the network parameterises r hat in eqn 1 from the paper
        """
        Compute the reward prediction for a given input and ensemble member.
        
        Parameters
        ----------
        x : array_like
            The input data for which the reward is to be predicted.
        member : int, optional
            The index of the ensemble member to use for prediction. If `-1`, the last member is used.

        Returns
        -------
        torch.Tensor
            The predicted reward for the input data by the specified ensemble member.
        """

        return self.ensemble[member](torch.from_numpy(x).float().to(device))


    def r_hat(self, x):  # ✅
        # they say they average the rewards from each member of the ensemble, but I think this only makes sense if the rewards are already normalized
        # but I don't understand how the normalization should be happening right now :(
        """
        Compute the average reward prediction from all ensemble members for a given input.
        
        Parameters
        ----------
        x : array_like
            The input data for which the reward is to be predicted.
        
        Returns
        -------
        float
            The average predicted reward for the input data by all ensemble members.
        """
        r_hats = []
        for member in range(self.de):
            r_hats.append(self.r_hat_member(x, member=member).detach().cpu().numpy())
        r_hats = np.array(r_hats)
        # np.mean(r_hats) computes the mean of all numbers in the array
        return np.mean(r_hats)  # Returns a scalar
    
    
    def r_hat_batch(self, x):  # ✅
        # they say they average the rewards from each member of the ensemble, but I think this only makes sense if the rewards are already normalized
        # but I don't understand how the normalization should be happening right now :(
        """
        Compute the average reward prediction from all ensemble members for a given batch of input data.
        
        Parameters
        ----------
        x : array_like
            The input data for which the reward is to be predicted.
        
        Returns
        -------
        array_like
            The average predicted rewards for the input data by all ensemble members.
        """
        r_hats = []
        for member in range(self.de):
            r_hats.append(self.r_hat_member(x, member=member).detach().cpu().numpy())
        r_hats = np.array(r_hats)

        return np.mean(r_hats, axis=0)
    
  
    def save(self, model_dir, step):  # ✅
        """
        Save the state dictionaries of all ensemble members to disc.
        
        Parameters
        ----------
        model_dir : str
            The directory where the model files will be saved.
        step : int
            The current training step or iteration, used for naming the saved model files.
        """

        for member in range(self.de):
            torch.save(
                self.ensemble[member].state_dict(), '%s/reward_model_%s_%s.pt' % (model_dir, step, member)
            )


    def load(self, model_dir, step):  # ✅
        """
        Load the state dictionaries of all ensemble members from disc.
        
        Parameters
        ----------
        model_dir : str
            The directory where the model files are saved.
        step : int
            The current training step or iteration, used for naming the saved model files.
        """
        for member in range(self.de):
            self.ensemble[member].load_state_dict(
                torch.load('%s/reward_model_%s_%s.pt' % (model_dir, step, member))
            )
       
            
    # This method is designed to evaluate the training accuracy of the reward model by comparing the predicted ranking (i.e. which of two trajectory segments is better) from each ensemble member with both the assigned labels (which may be “soft” or derived via teacher signals) and the ground-truth labels. The method processes the stored training data in batches from internal buffers.
    
    # To compute the accuracy of the reward predictions (ranking of two segments) for each ensemble member and return the mean accuracy across the ensemble. It calculates two metrics:
    # ensemble_acc: Accuracy with respect to the assigned labels.
    # gt_ensemble_acc: Accuracy with respect to the ground-truth labels
    def get_train_acc(self):  # ✅
        # ensemble_acc will accumulate the number of correct predictions (matches between the predicted ranking and the assigned labels) for each ensemble member.
        # gt_ensemble_acc will do the same for the ground-truth labels
        ensemble_acc = np.array([0 for _ in range(self.de)])
        gt_ensemble_acc = np.array([0 for _ in range(self.de)])
        # This value (max_len) is used to know how many stored samples (or trajectory pairs) are available for evaluation.
        max_len = self.capacity if self.buffer_full else self.buffer_index
        # Randomizing the order helps ensure that the evaluation is not biased by the order in which data was stored
        total_batch_index = np.random.permutation(max_len)
        # Processing data in batches prevents memory overload and enables vectorized operations for speed
        batch_size = 256
        num_epochs = int(np.ceil(max_len/batch_size))
        
        # This counter keeps track of the total number of samples (or transitions) processed across all batches. It is used later to normalize the accumulated correct predictions into an accuracy value.
        total = 0
        # for each batch
        for epoch in range(num_epochs):
            # The code calculates the ending index for the current batch. If the computed ending index exceeds the number of available samples, it is capped at max_len. This ensures that we do not try to access data beyond what is available.
            last_index = (epoch+1)*batch_size
            if (epoch+1)*batch_size > max_len:
                last_index = max_len
            
            # slice the stored data - per batch    
            sa_t_1 = self.buffer_seg1[epoch*batch_size:last_index]
            sa_t_2 = self.buffer_seg2[epoch*batch_size:last_index]
            labels = self.buffer_label[epoch*batch_size:last_index]
            gt_labels = self.gt_buffer_label[epoch*batch_size:last_index]
            labels = torch.from_numpy(labels.flatten()).long().to(device)
            gt_labels = torch.from_numpy(gt_labels.flatten()).long().to(device)

            total += labels.size(0)  # Accumulate the total number of samples processed
            for member in range(self.de):  # for each ensemble member
                # These predictions represent the estimated rewards for the two trajectory segments.
                r_hat1 = self.r_hat_member(sa_t_1, member=member)
                r_hat2 = self.r_hat_member(sa_t_2, member=member)
                # The aggregated reward for each segment is then used to determine which segment is preferred.
                r_hat1 = r_hat1.sum(axis=1)
                r_hat2 = r_hat2.sum(axis=1)
                # This structure is required so that a softmax can be applied to obtain a probability distribution over the two segments.
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)  
                # This predicted index is the model’s ranking decision for the current sample.              
                _, predicted = torch.max(r_hat.data, 1)
                # These counts measure how well the ensemble member’s predictions align with both sets of labels.
                correct = (predicted == labels).sum().item()
                gt_correct = (predicted == gt_labels).sum().item()
                # The correct predictions for the current batch are added to the cumulative totals for the ensemble member.
                ensemble_acc[member] += correct
                gt_ensemble_acc[member] += gt_correct
        # These values represent the performance of each ensemble member across the entire training dataset.        
        ensemble_acc = ensemble_acc / total
        gt_ensemble_acc = gt_ensemble_acc / total
        # summary measure of how well the reward model is performing.
        return np.mean(ensemble_acc), np.mean(gt_ensemble_acc)
    
    
    # This method randomly selects a mini‐batch of trajectory segments from the stored training data (contained in self.inputs and self.targets) and then “samples” a specific time window from each trajectory.
    def get_queries(self, mb_size=20): # ✅
        # len_traj is set to the length of the first stored trajectory (i.e. the number of transitions in that trajectory). max_len is the total number of stored trajectories in self.inputs.
        len_traj, max_len = len(self.inputs[0]), len(self.inputs)
        
        # If the most recent trajectory is incomplete (i.e. its length is less than that of a full trajectory), then it is not used.
        if len(self.inputs[-1]) < len_traj:
            max_len = max_len - 1
        
        # convert to numpy arrays
        train_inputs = np.array(self.inputs[:max_len])
        train_targets = np.array(self.targets[:max_len])

        # For each query, two sets of trajectories are sampled independently: batch_index_2 randomly selects mb_size trajectories for the second segment. batch_index_1 does the same for the first segment. The corresponding trajectory segments (sa_t_1 and sa_t_2) and their reward sequences (r_t_1 and r_t_2) are obtained
        batch_index_2 = np.random.choice(max_len, size=mb_size, replace=True)
        sa_t_2 = train_inputs[batch_index_2] # Batch x T x dim of s&a
        r_t_2 = train_targets[batch_index_2] # Batch x T x 1
        
        batch_index_1 = np.random.choice(max_len, size=mb_size, replace=True)
        sa_t_1 = train_inputs[batch_index_1] # Batch x T x dim of s&a
        r_t_1 = train_targets[batch_index_1] # Batch x T x 1
         
        # The trajectories are reshaped such that the batch and time dimensions are merged. The result is a 2D array where each row corresponds to one transition, and there are (Batch x T) rows. This flattening is useful because the next step will extract a specific window (time index) from these flattened sequences.       
        sa_t_1 = sa_t_1.reshape(-1, sa_t_1.shape[-1]) # (Batch x T) x dim of s&a
        r_t_1 = r_t_1.reshape(-1, r_t_1.shape[-1]) # (Batch x T) x 1
        sa_t_2 = sa_t_2.reshape(-1, sa_t_2.shape[-1]) # (Batch x T) x dim of s&a
        r_t_2 = r_t_2.reshape(-1, r_t_2.shape[-1]) # (Batch x T) x 1

        # To extract a window (segment) of the trajectory for each query, allowing variability in which part of the trajectory is used.
        time_index = np.array([list(range(i*len_traj,
                                            i*len_traj+self.size_segment)) for i in range(mb_size)])
        time_index_2 = time_index + np.random.choice(len_traj-self.size_segment, size=mb_size, replace=True).reshape(-1,1)
        time_index_1 = time_index + np.random.choice(len_traj-self.size_segment, size=mb_size, replace=True).reshape(-1,1)
        
        # np.take is used to select rows from the flattened arrays based on the computed time indices. After this operation, each output (sa_t_1, sa_t_2, r_t_1, r_t_2) has the shape [Batch, size_seg, ...].
        sa_t_1 = np.take(sa_t_1, time_index_1, axis=0) # Batch x size_seg x dim of s&a
        r_t_1 = np.take(r_t_1, time_index_1, axis=0) # Batch x size_seg x 1
        sa_t_2 = np.take(sa_t_2, time_index_2, axis=0) # Batch x size_seg x dim of s&a
        r_t_2 = np.take(r_t_2, time_index_2, axis=0) # Batch x size_seg x 1
                
        return sa_t_1, sa_t_2, r_t_1, r_t_2


    # This method stores the query pairs (and their labels) into circular buffers. It handles buffer overflow with a FIFO (First-In-First-Out) mechanism.
    def put_queries(self, sa_t_1, sa_t_2, labels, gt_labels, r_t_1=None, r_t_2=None, fake_traj=False):  # ✅
        # Determine the number of samples to insert
        total_sample = sa_t_1.shape[0]
        # origin_index is the current position in the buffer where new data will be inserted.
        origin_index = self.buffer_index
        # next_index is the index where insertion would end if all samples fit.
        next_index = self.buffer_index + total_sample
        # If the new data would exceed the buffer’s capacity:
        if next_index >= self.capacity:
            # Set a flag self.buffer_full to indicate that the buffer has wrapped around.
            self.buffer_full = True
            # Calculate maximum_index, which is how many samples can be stored in the current remaining space.
            maximum_index = self.capacity - self.buffer_index
            # Use np.copyto to fill in the remaining slots in the buffer from self.buffer_index up to self.capacity
            np.copyto(self.buffer_seg1[self.buffer_index:self.capacity], sa_t_1[:maximum_index])
            np.copyto(self.buffer_seg2[self.buffer_index:self.capacity], sa_t_2[:maximum_index])
            np.copyto(self.buffer_label[self.buffer_index:self.capacity], labels[:maximum_index])
            np.copyto(self.gt_buffer_label[self.buffer_index:self.capacity], gt_labels[:maximum_index])
            # The fake_flag for these indices is also set.
            self.fake_flag[self.buffer_index:self.capacity] = fake_traj
            # Handle the Remainder by Wrapping Around
            remain = total_sample - (maximum_index)
            if remain > 0:
                np.copyto(self.buffer_seg1[0:remain], sa_t_1[maximum_index:])
                np.copyto(self.buffer_seg2[0:remain], sa_t_2[maximum_index:])
                np.copyto(self.buffer_label[0:remain], labels[maximum_index:])
                np.copyto(self.gt_buffer_label[0:remain], gt_labels[maximum_index:])
                self.fake_flag[0:remain] = fake_traj
            # The new buffer_index is set to remain (i.e., the position following the wrapped-around data).
            self.buffer_index = remain
        else:
            # If there is enough space in the buffer (i.e., next_index < self.capacity), then simply insert the data from the current buffer index to next_index. Update self.buffer_index to next_index after insertion.
            np.copyto(self.buffer_seg1[self.buffer_index:next_index], sa_t_1)
            np.copyto(self.buffer_seg2[self.buffer_index:next_index], sa_t_2)
            np.copyto(self.buffer_label[self.buffer_index:next_index], labels)
            np.copyto(self.gt_buffer_label[self.buffer_index:next_index], gt_labels)
            self.fake_flag[self.buffer_index:next_index] = fake_traj
            self.buffer_index = next_index
     
     
    # assign labels (or preferences) to a pair of trajectory segments based on their predicted rewards       
    def get_label(self, sa_t_1, sa_t_2, r_t_1, r_t_2):  # ✅
        # Begin by saving the current buffer index (used later for saving files if needed).
        origin_index = self.buffer_index
        # For each query (each pair of trajectory segments), compute the total reward by summing across the time dimension (axis 1) for both segments.
        sum_r_t_1 = np.sum(r_t_1, axis=1)
        sum_r_t_2 = np.sum(r_t_2, axis=1)
        
        # skip the query - The idea is that if neither segment in a query is “challenging” enough (i.e. both have low rewards), the query might not be informative. If no query meets the threshold, the method returns early with None values.
        if self.teacher_thres_skip > 0: # Checks if there is a nonzero threshold (teacher_thres_skip) to decide whether to skip a query.
            max_r_t = np.maximum(sum_r_t_1, sum_r_t_2)
            # Creates a Boolean mask (max_index) indicating which queries have a maximum reward exceeding this threshold.
            max_index = (max_r_t > self.teacher_thres_skip).reshape(-1)
            if sum(max_index) == 0:
                return None, None, None, None, []

            # Filters the query pairs and their reward segments to include only those that meet the threshold. This focus on “informative” queries helps the training process by discarding cases that may not provide a clear preference signal.
            sa_t_1 = sa_t_1[max_index]
            sa_t_2 = sa_t_2[max_index]
            r_t_1 = r_t_1[max_index]
            r_t_2 = r_t_2[max_index]
            sum_r_t_1 = np.sum(r_t_1, axis=1)
            sum_r_t_2 = np.sum(r_t_2, axis=1)
        
        # equally preferable - Creates a Boolean mask (margin_index) for queries where the absolute difference between the two cumulative rewards is less than a set threshold (teacher_thres_equal). If the rewards are too similar, the query might be ambiguous. In such cases, the label will later be set to a special value (–1) to indicate “equal preference.”
        margin_index = (np.abs(sum_r_t_1 - sum_r_t_2) < self.teacher_thres_equal).reshape(-1)
        
        # perfectly rational
        # Copies the reward arrays into temporary arrays.
        seg_size = r_t_1.shape[1]
        temp_r_t_1 = r_t_1.copy()
        temp_r_t_2 = r_t_2.copy()
        # Applies a decay factor (teacher_gamma) multiplicatively to earlier portions of the trajectory. The loop multiplies the rewards for the first index+1 time steps by teacher_gamma for each index. This “decay” can simulate the idea that later rewards might be more relevant (or vice versa), making the label “perfectly rational” according to a specific hypothesis or teacher model.
        for index in range(seg_size-1):
            temp_r_t_1[:,:index+1] *= self.teacher_gamma
            temp_r_t_2[:,:index+1] *= self.teacher_gamma
        sum_r_t_1 = np.sum(temp_r_t_1, axis=1)
        sum_r_t_2 = np.sum(temp_r_t_2, axis=1)
          
        # Compares the adjusted cumulative rewards:If the sum for the first trajectory is less than the second, it assigns a label 1 (meaning, for example, that the first trajectory is preferable). Otherwise, it assigns 0.  
        rational_labels = 1*(sum_r_t_1 < sum_r_t_2)
        # If teacher_beta is positive, the method uses a Bradley-Terry model—a probabilistic model for paired comparisons: It creates a tensor r_hat by concatenating the two cumulative rewards. Scales it by teacher_beta. Applies softmax to obtain a probability distribution, then takes the probability associated with the second element (i.e. the probability that x_2 is preferred, so 1 minus that probability gives the chance that x_1 is preferred). Samples a binary outcome from a Bernoulli distribution using this probability. If teacher_beta is not positive, it simply uses the rational_labels computed earlier.
        if self.teacher_beta > 0: # Bradley-Terry rational model
            r_hat = torch.cat([torch.Tensor(sum_r_t_1), 
                            torch.Tensor(sum_r_t_2)], axis=-1)
            r_hat = r_hat*self.teacher_beta
            ent = F.softmax(r_hat, dim=-1)[:, 1]
            labels = torch.bernoulli(ent).int().numpy().reshape(-1, 1)
        else:
            labels = rational_labels
        
        # Generates random numbers for each label.
        # Identifies which labels should be “flipped” based on a probability teacher_eps_mistake.
        # Flips those labels (i.e., 0 becomes 1, and 1 becomes 0). This simulates the idea that the teacher (or LLM) might sometimes make mistakes, adding realism or robustness to the training process.
        len_labels = labels.shape[0]
        rand_num = np.random.rand(len_labels)
        noise_index = rand_num <= self.teacher_eps_mistake
        labels[noise_index] = 1 - labels[noise_index]

        # equally preferable - The value -1 indicates that the two trajectories are considered equally preferable, reflecting ambiguity in the decision.
        labels[margin_index] = -1 

        gt_labels = labels # - The computed labels are also stored as ground-truth labels (gt_labels). These may later be used for evaluation or comparison with labels obtained from an LLM.
        # else - If the flag self.vlm_label is set, the method calls get_vlm_label to potentially override or refine the labels using feedback from a language model.
        if self.vlm_label:
            labels = self.get_vlm_label(sa_t_1, sa_t_2, r_t_1, r_t_2, labels)

        return sa_t_1, sa_t_2, r_t_1, r_t_2, labels, gt_labels
    

    # This method uses a language model (via GPT inference) to obtain labels for each query pair. It can serve as an override or double-check for the labels computed using the reward sums
    def get_vlm_label(self, sa_t_1, sa_t_2, r_t_1, r_t_2, labels): # ✅
        # Captures the current buffer index for use in saving data to disk later.
        origin_index = self.buffer_index
        # Methods for gpt4 inference
        from vlms.gpt4_infer import gpt_infer, gpt_infer_traj_gen, gpt_infer_traj_gen_check

        # Create an array with the same shape as `labels`, initialized with -1.
        # As the method processes each query, valid labels will overwrite these initial values.
        vlm_labels = np.full(labels.shape, -1, dtype=int)

        for idx in range(sa_t_1.shape[0]): # Foreach query pair
            # Logging: Prints a message indicating that it is processing query idx.
            print("querying {} {}/{}".format(self.vlm_label, idx, sa_t_1.shape[0]))
            # Calls traj_pair_process to process the trajectory pair into a structured format and obtain two processed trajectories (traj_1 and traj_2).
            data_to_save, traj_1, traj_2 = traj_pair_process(self.env_name, sa_t_1[idx], sa_t_2[idx], r_t_1[idx], r_t_2[idx], labels[idx])
            # Uses format_traj_for_gpt to convert the trajectory data into a text string (traj_str) suitable for a GPT prompt.
            traj_str = format_traj_for_gpt(traj_1, traj_2)
            # Retrieves a template prompt from gpt_query_env_prompts for the specific environment, and formats it by inserting the segment size and the trajectory string
            query_prompt = gpt_query_env_prompts[self.env_name]
            query_prompt = query_prompt.format(self.size_segment, traj_str)

            # Calls gpt_infer with the label type (self.vlm_label) and the constructed prompt.
            # Processes the raw response using get_response_answer to extract a clean answer.
            res = gpt_infer(self.vlm_label, query_prompt)
            answer = get_response_answer(res)

            # Attempts to convert the answer (a string) into an integer.
            # If the answer is 1 or 2, it subtracts 1, so the labels become 0 or 1.
            # If the answer is anything else or if conversion fails, the label is set to -1, indicating an invalid response.
            try:
                # Try to convert the answer to an integer
                label_res = int(answer)
                # If `label_res` is 1 or 2, subtract 1
                if label_res == 1 or label_res == 2:
                    label_res = label_res - 1
                else:
                    # Otherwise, set it to -1
                    label_res = -1
            except:
                # If any exception occurs, set it to -1
                label_res = -1

            # This step aims to verify the consistency of the LLM’s output. If the model is inconsistent upon swapping, the label is considered unreliable.
            if self.double_check and label_res != -1:
                # Perform double-check logic by swapping `traj_1` and `traj_2`
                traj_str_swapped = format_traj_for_gpt(traj_2, traj_1)
                query_prompt_swapped = gpt_query_env_prompts[self.env_name].format(self.size_segment, traj_str_swapped)

                # Call GPT inference again
                res_swapped = gpt_infer(self.vlm_label, query_prompt_swapped)
                answer_swapped = get_response_answer(res_swapped)

                try:
                    # Try to convert the swapped answer to an integer
                    label_res_swapped = int(answer_swapped)
                    # If `label_res_swapped` is 1 or 2, subtract 1
                    if label_res_swapped == 1 or label_res_swapped == 2:
                        label_res_swapped = label_res_swapped - 1
                    else:
                        label_res_swapped = -1
                except:
                    label_res_swapped = -1

                # If the two predictions are not symmetric, discard the label and set it to -1
                if not ((label_res == 0 and label_res_swapped == 1) or (label_res == 1 and label_res_swapped == 0)):
                    print("Double check False!")
                    label_res = -1

            # Fill the result into the corresponding position of `vlm_labels`
            vlm_labels[idx] = label_res

            # Save `data_to_save` and `res` to JSON files
            # If a save path is provided, creates the directory if it doesn’t exist.
            # Saves both the processed trajectory pair (data_to_save) and the raw LLM response (res) to JSON files. - For debugging.
            if self.traj_save_path is not None:
                # Create directory if it doesn't exist
                os.makedirs(self.traj_save_path, exist_ok=True)
                # Save `data_to_save`
                data_file_path = os.path.join(self.traj_save_path, f"traj_pairs_{origin_index+idx}.json")
                with open(data_file_path, 'w') as data_file:
                    json.dump(data_to_save, data_file, indent=4)
                # Save `res` (inference results)
                res_file_path = os.path.join(self.traj_save_path, f"llm_response_{origin_index+idx}.json")
                with open(res_file_path, 'w') as res_file:
                    json.dump(res, res_file, default=lambda o: o.__dict__, indent=4)

            # If better_traj_gen is enabled and the label is valid, it attempts to generate an improved trajectory:
            # if self.better_traj_gen and (label_res != -1):
            #     print("querying better_traj_gen {} {}/{}".format(self.vlm_label, idx, sa_t_1.shape[0]))
            #     # Select the better trajectory
            #     traj_better = traj_1 if label_res == 0 else traj_2

            #     if "metaworld" in self.env_name:
            #         traj_better_begin = {
            #             'tcp': traj_better['tcp'][0],
            #             'obj': traj_better['obj'][0],
            #             'target': traj_better['target']  # Keep all `target` elements
            #         }
            #     elif "PointMaze" in self.env_name:
            #         traj_better_begin = {
            #             'position': traj_better['position'][0],
            #             'target': traj_better['target']  # Keep all `target` elements
            #         }

            #     traj_better_begin_str = format_traj_begin_for_gpt(traj_better_begin)
            #     trajgen_prompt = trajgen_template(self.env_name).format(self.size_segment, traj_better_begin_str)
            #     res_traj = gpt_infer_traj_gen(self.vlm_label, query_prompt, res, trajgen_prompt)
            #     new_traj_dist = extract_trajectory_from_text(self.env_name, res_traj, self.size_segment)

                # Save `res_traj` (inference results)
                # if self.traj_save_path is not None:
                #     res_traj_file_path = os.path.join(self.traj_save_path, f"llm_trajgen_{origin_index+idx}.json")
                #     with open(res_traj_file_path, 'w') as res_traj_file:
                #         json.dump(res_traj, res_traj_file, default=lambda o: o.__dict__, indent=4)

                #     # Select the worse trajectory
                #     worse_traj = sa_t_1[idx] if label_res == 0 else sa_t_2[idx]
                #     better_traj = convert_trajdist_to_traj(self.env_name, worse_traj, new_traj_dist)

                #     # Reshape to add one dimension
                #     worse_traj = worse_traj.reshape(1, *worse_traj.shape)
                #     better_traj = better_traj.reshape(1, *better_traj.shape)
                #     self.put_queries(worse_traj, better_traj, labels=np.array([[1]]), gt_labels=np.array([[1]]), fake_traj=True)

                #     print("Valid trajectory, Store it to buffer...")
                # else:
                #     print("Invalid trajectory, discarding...")

        # Filter out parts of `vlm_labels` with -1
        # It filters out queries that were deemed invalid (labels still equal to -1).
        # If there are valid predictions, it compares the LLM-generated labels (filtered_vlm_labels) with the original target labels (filtered_labels) to compute an accuracy.
        # Prints the computed accuracy.
        valid_indices = [i for i, label in enumerate(vlm_labels) if label != -1]
        if valid_indices:  # If there are valid predictions
            filtered_vlm_labels = vlm_labels[valid_indices]
            filtered_labels = labels[valid_indices]
            # Calculate accuracy
            correct = sum(1 for vlm_label, target_label in zip(filtered_vlm_labels, filtered_labels) if vlm_label == target_label)
            accuracy = correct / len(filtered_vlm_labels)
            print(f"LLM Label Accuracy: {accuracy * 100:.2f}%")
        else:
            print("No valid predictions to compute accuracy.")
            accuracy = 0.0  # If no valid predictions, set accuracy to 0

        # Stores the computed LLM query accuracy in the class attribute.
        # Returns the array of LLM labels.
        # helps monitor the quality of LLM feedback. 
        self.llm_query_accuracy = accuracy

        return vlm_labels



    # ****************************************************************************
    # When implemented fully, its goal would be to return a number of selected queries (or their labels) that have been chosen solely based on diversity using the k-center approach
    def kcenter_sampling(self): # ****
        return len(labels)
    
    # The k-center-based methods (including the variants that incorporate disagreement or entropy) aim to choose a diverse subset of queries. This helps ensure that the training data covers different regions of the input space.
    def kcenter_disagree_sampling(self): # ****
        # num_init: The initial number of queries is determined by multiplying the mini-batch size (mb_size) by a scaling factor (large_batch). num_init_half: Half of that number will be used for further selection
        num_init = self.mb_size*self.large_batch
        num_init_half = int(num_init*0.5)
        
        # get queries - Retrieves a batch of query pairs (trajectory segments and their rewards) using the get_queries method.
        sa_t_1, sa_t_2, r_t_1, r_t_2 =  self.get_queries(
            mb_size=num_init)
        
        # get final queries based on uncertainty - Calls get_rank_probability to obtain, among other things, the standard deviation (disagreement) across ensemble predictions.
        # Sorts the disagreement values in descending order (by negating and then argsorting) and selects the top half (num_init_half) of queries that are the most uncertain.
        _, disagree = self.get_rank_probability(sa_t_1, sa_t_2)
        top_k_index = (-disagree).argsort()[:num_init_half]
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]
        
        # get final queries based on kmeans clustering
        temp_sa_t_1 = sa_t_1[:,:,:self.ds]
        temp_sa_t_2 = sa_t_2[:,:,:self.ds]
        temp_sa = np.concatenate([temp_sa_t_1.reshape(num_init_half, -1),  
                                  temp_sa_t_2.reshape(num_init_half, -1)], axis=1)
        
        
        max_len = self.capacity if self.buffer_full else self.buffer_index
        
        tot_sa_1 = self.buffer_seg1[:max_len, :, :self.ds]
        tot_sa_2 = self.buffer_seg2[:max_len, :, :self.ds]
        tot_sa = np.concatenate([tot_sa_1.reshape(max_len, -1),  
                                 tot_sa_2.reshape(max_len, -1)], axis=1)
        
        selected_index = KCenterGreedy(temp_sa, tot_sa, self.mb_size)
        
        r_t_1, sa_t_1 = r_t_1[selected_index], sa_t_1[selected_index]
        r_t_2, sa_t_2 = r_t_2[selected_index], sa_t_2[selected_index]

        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels, gt_labels = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2)
        
        if len(labels) > 0:
            if self.traj_save_path is not None:
                self.put_queries(sa_t_1, sa_t_2, labels, gt_labels, r_t_1, r_t_2)
            else:
                self.put_queries(sa_t_1, sa_t_2, gt_labels, labels)
        
        return len(labels)
    
    # This method is very similar to kcenter_disagree_sampling, except that it uses entropy (a measure of uncertainty) instead of disagreement from the ensemble.
    def kcenter_entropy_sampling(self): # ****
        
        num_init = self.mb_size*self.large_batch
        num_init_half = int(num_init*0.5)
        
        # get queries
        sa_t_1, sa_t_2, r_t_1, r_t_2 =  self.get_queries(
            mb_size=num_init)
        
        
        # get final queries based on uncertainty
        entropy, _ = self.get_entropy(sa_t_1, sa_t_2)
        top_k_index = (-entropy).argsort()[:num_init_half]
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]
        
        # get final queries based on kmeans clustering
        temp_sa_t_1 = sa_t_1[:,:,:self.ds]
        temp_sa_t_2 = sa_t_2[:,:,:self.ds]
        
        temp_sa = np.concatenate([temp_sa_t_1.reshape(num_init_half, -1),  
                                  temp_sa_t_2.reshape(num_init_half, -1)], axis=1)
        
        max_len = self.capacity if self.buffer_full else self.buffer_index
        
        tot_sa_1 = self.buffer_seg1[:max_len, :, :self.ds]
        tot_sa_2 = self.buffer_seg2[:max_len, :, :self.ds]
        tot_sa = np.concatenate([tot_sa_1.reshape(max_len, -1),  
                                 tot_sa_2.reshape(max_len, -1)], axis=1)
        
        selected_index = KCenterGreedy(temp_sa, tot_sa, self.mb_size)
        
        r_t_1, sa_t_1 = r_t_1[selected_index], sa_t_1[selected_index]
        r_t_2, sa_t_2 = r_t_2[selected_index], sa_t_2[selected_index]

        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2)
        
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
        
        return len(labels)
    
    # Instead of using uncertainty or diversity measures, it simply randomly samples queries with a batch size equal to mb_size. - Randomly selects queries without any additional criteria. This is the simplest approach and can serve as a baseline.
    def uniform_sampling(self): # ✅ 
        # get queries
        sa_t_1, sa_t_2, r_t_1, r_t_2 =  self.get_queries(
            mb_size=self.mb_size)
            
        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels, gt_labels = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2)
        
        if not self.save_equal:
                valid_indices = labels[:, 0] != -1
                sa_t_1 = sa_t_1[valid_indices]
                sa_t_2 = sa_t_2[valid_indices]
                r_t_1 = r_t_1[valid_indices]
                r_t_2 = r_t_2[valid_indices]
                labels = labels[valid_indices]
                gt_labels = gt_labels[valid_indices]

        if len(labels) > 0:
            if not (self.vlm_label and  (not self.vlm_feedback) and self.better_traj_gen):
                if self.traj_save_path is not None:
                    self.put_queries(sa_t_1, sa_t_2, labels, gt_labels, r_t_1, r_t_2)
                else:
                    self.put_queries(sa_t_1, sa_t_2, gt_labels, labels)
        
        return len(labels)
    
    # These methods select queries where the ensemble is uncertain (i.e. high disagreement or high entropy). They are useful if you want the model to focus on samples where it is unsure, which can be very informative for training.
    def disagreement_sampling(self):
        
        # get queries
        sa_t_1, sa_t_2, r_t_1, r_t_2 =  self.get_queries(
            mb_size=self.mb_size*self.large_batch)
        
        # get final queries based on uncertainty
        _, disagree = self.get_rank_probability(sa_t_1, sa_t_2)
        top_k_index = (-disagree).argsort()[:self.mb_size]
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]        
        
        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2)        
        
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
        
        return len(labels)
    
    def entropy_sampling(self):
        
        # get queries
        sa_t_1, sa_t_2, r_t_1, r_t_2 =  self.get_queries(
            mb_size=self.mb_size*self.large_batch)
        
        # get final queries based on uncertainty
        entropy, _ = self.get_entropy(sa_t_1, sa_t_2)
        
        top_k_index = (-entropy).argsort()[:self.mb_size]
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]
        
        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(    
            sa_t_1, sa_t_2, r_t_1, r_t_2)
        
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
        
        return len(labels)
    
    
    def train_reward(self):
        # The model is an ensemble with self.de members. Each member is trained on its own randomly permuted version of the training indices.
        # A list of lists to store the loss for each ensemble member over training.
        ensemble_losses = [[] for _ in range(self.de)] 
        # Numpy arrays to count correct predictions overall and on “valid” (non-fake) examples, respectively.
        ensemble_acc = np.array([0 for _ in range(self.de)])
        gt_ensemble_acc = np.array([0 for _ in range(self.de)])
        # Determines the number of data points to use (either full capacity or current index).
        max_len = self.capacity if self.buffer_full else self.buffer_index
        # For each ensemble member, generate a random permutation of indices to randomize the training order.
        total_batch_index = []
        for _ in range(self.de):
            total_batch_index.append(np.random.permutation(max_len))
        num_epochs = int(np.ceil(max_len / self.train_batch_size))
        # Used to accumulate the total number of examples and the number of valid (non-fake) examples, respectively, for accuracy calculations.
        total = 0
        gt_total = 0

        for epoch in range(num_epochs):
            # Zero out the gradients before each epoch.
            self.opt.zero_grad()
            loss = 0.0
            # Calculate the index range for the current batch; if it exceeds max_len, clamp it.
            last_index = (epoch + 1) * self.train_batch_size
            if last_index > max_len:
                last_index = max_len

            #  extract a mini-batch of indices and use them to retrieve: Two segments (sa_t_1 and sa_t_2) Predicted labels and ground truth labels. A flag array (fake_flag) that indicates if a label should be considered “fake.”
            for member in range(self.de):
                # get random batch
                idxs = total_batch_index[member][epoch * self.train_batch_size:last_index]
                sa_t_1 = self.buffer_seg1[idxs]
                sa_t_2 = self.buffer_seg2[idxs]
                labels = self.buffer_label[idxs]
                gt_labels = self.gt_buffer_label[idxs]
                fake_labels = self.fake_flag[idxs]
                # Convert Labels to Tensors:
                labels = torch.from_numpy(labels.flatten()).long().to(device)
                gt_labels = torch.from_numpy(gt_labels.flatten()).long().to(device)
                # Identify the indices where the labels are valid
                # Note: Do not filter during training, only for label accuracy calculation
                valid_indices = np.where(fake_labels.flatten() == False)[0]
                if member == 0:
                    total += labels.size(0)
                    gt_total += len(valid_indices)
                # get logits - Compute Model Predictions (Logits):
                r_hat1 = self.r_hat_member(sa_t_1, member=member)
                r_hat2 = self.r_hat_member(sa_t_2, member=member)
                r_hat1 = r_hat1.sum(axis=1)
                r_hat2 = r_hat2.sum(axis=1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)
                # compute loss - The standard cross-entropy loss is computed between the logits and the true labels.
                curr_loss = self.CEloss(r_hat, labels)
                loss += curr_loss
                ensemble_losses[member].append(curr_loss.item())
                # compute acc for all data
                _, predicted = torch.max(r_hat.data, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct
                # Accumulate accuracy on all data
                if len(valid_indices) > 0:
                    gt_correct = (predicted[valid_indices] == gt_labels[valid_indices]).sum().item()
                    gt_ensemble_acc[member] += gt_correct
                else:
                    gt_ensemble_acc[member] += 0
            # Backpropagate the total loss.
            # Use the optimizer (self.opt) to update model parameters.
            loss.backward()
            self.opt.step()
        # Final Accuracy Computation and Return:
        ensemble_acc = ensemble_acc / total
        if gt_total != 0:
            self.llm_label_accuracy = gt_ensemble_acc / gt_total
        else:
            self.llm_label_accuracy = 0 

        return self.llm_query_accuracy, self.llm_label_accuracy, ensemble_acc

    # Same as above except - 
    # When labels equal -1 (indicating uniform or uncertain cases), they are replaced with a default value (0) and then soft targets are constructed. Loss Function: Uses a soft cross-entropy loss (softXEnt_loss) rather than the standard CE loss
    def train_soft_reward(self):
        ensemble_losses = [[] for _ in range(self.de)]
        ensemble_acc = np.array([0 for _ in range(self.de)])
        gt_ensemble_acc = np.array([0 for _ in range(self.de)])
        
        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = []
        for _ in range(self.de):
            total_batch_index.append(np.random.permutation(max_len))
        
        num_epochs = int(np.ceil(max_len/self.train_batch_size))
        # list_debug_loss1, list_debug_loss2 = [], []
        total = 0
        
        for epoch in range(num_epochs):
            self.opt.zero_grad()
            loss = 0.0
            
            last_index = (epoch+1)*self.train_batch_size
            if last_index > max_len:
                last_index = max_len
                
            for member in range(self.de):
                
                # get random batch
                idxs = total_batch_index[member][epoch*self.train_batch_size:last_index]
                sa_t_1 = self.buffer_seg1[idxs]
                sa_t_2 = self.buffer_seg2[idxs]
                labels = self.buffer_label[idxs]
                gt_labels = self.gt_buffer_label[idxs]
                labels = torch.from_numpy(labels.flatten()).long().to(device)
                gt_labels = torch.from_numpy(gt_labels.flatten()).long().to(device)
                
                if member == 0:
                    total += labels.size(0)
                
                # get logits
                r_hat1 = self.r_hat_member(sa_t_1, member=member)
                r_hat2 = self.r_hat_member(sa_t_2, member=member)
                r_hat1 = r_hat1.sum(axis=1)
                r_hat2 = r_hat2.sum(axis=1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)

                # compute loss
                uniform_index = labels == -1
                labels[uniform_index] = 0
                target_onehot = torch.zeros_like(r_hat).scatter(1, labels.unsqueeze(1), self.label_target)
                target_onehot += self.label_margin
                if sum(uniform_index) > 0:
                    target_onehot[uniform_index] = 0.5
                curr_loss = self.softXEnt_loss(r_hat, target_onehot)
                loss += curr_loss
                ensemble_losses[member].append(curr_loss.item())
                
                # compute acc
                _, predicted = torch.max(r_hat.data, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct
                gt_correct = (predicted == gt_labels).sum().item()
                gt_ensemble_acc[member] += gt_correct
                
            loss.backward()
            self.opt.step()
        
        ensemble_acc = ensemble_acc / total
        self.llm_label_accuracy = gt_ensemble_acc / total
        
        return self.llm_query_accuracy, self.llm_label_accuracy, ensemble_acc