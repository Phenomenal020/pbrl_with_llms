# Source: https://github.com/TU2021/RL-SaLLM-F/blob/main/replay_buffer.py


import numpy as np
import torch

class ReplayBuffer(object):
    """Buffer to store environment transitions."""


    def __init__(self, obs_shape, action_shape, capacity, device):
        """
        :param obs_shape: Shape of observations.
        :param action_shape: Shape of actions.
        :param capacity: How many transitions to store in the buffer.
        :param device: Device where the buffer is stored.
        """
        self.capacity = capacity
        self.device = device

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False


    def __len__(self):
        """ 
        Returns the number of elements currently in the buffer. If the buffer is full,
        this will be equal to the buffer's capacity, otherwise it will be the index of
        the next element to be added.
        """
        return self.capacity if self.full else self.idx


    def add(self, obs, action, reward, next_obs, done, done_no_max):
        """
        Adds a new transition to the buffer.
        It decides where to add the transition based on the current index
        and whether the buffer is full.
        
        Parameters
        ----------
        obs : array_like
            The observation at the current timestep.
        action : array_like
            The action taken at the current timestep.
        reward : float
            The reward received at the next timestep.
        next_obs : array_like
            The observation at the next timestep.
        done : bool
            Whether the episode terminated at the next timestep.
        done_no_max : bool
            Whether the episode terminated at the next timestep, disregarding 
            the `max_episode_steps` limit.
        """
        np.copyto(self.obses[self.idx], obs) # copy to buffer using numpy's copyto method which is faster than np.copy
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.not_dones_no_max[self.idx], not done_no_max)

        # update index and full flag (circle buffer)
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    # TODO: Implement add batch if necessary


    def relabel_with_predictor(self, predictor):
        """
        Loops through the buffer and relabels the rewards in the buffer with predicted rewards
        from the reward model. This ensures that the rewards are consistent with the reward model's 
        predictions. In turn, the reward model should be constantly trained with new preferences.

        Parameters
        ----------
        predictor : Predictor
            The reward model

        Returns
        -------
        str
            A message indicating whether the relabeling process was successful.
        """        
        batch_size = 200
        total_iter = int(self.idx/batch_size) # Number of full batches
        if self.idx > batch_size*total_iter:
            total_iter += 1  # round up to cover partial batch
            
        for index in range(total_iter):  # for each batch
            last_index = (index+1)*batch_size # Get last index
            if (index+1)*batch_size > self.idx:
                last_index = self.idx # Adjust for partial batch eg, 100 instead of 200

            # Get observations corresponding to the batch    
            obses = self.obses[index*batch_size:last_index] 
            inputs = obses
            # Get predicted rewards from predictor on the observations and/or actions
            pred_reward = predictor.r_hat_batch(inputs) 
            # Replace rewards with these new predicted rewards
            self.rewards[index*batch_size:last_index] = pred_reward  

        print("Relabeled rewards with updated predictor successfully.")


    def sample(self, batch_size):
        """
        Sample a batch of transitions from the buffer.

        Parameters
        ----------
        batch_size : int
            The size of the batch to sample.

        Returns
        -------
        obses : torch.Tensor
            The observations at the current timesteps.
        actions : torch.Tensor
            The actions taken at the current timesteps.
        rewards : torch.Tensor
            The rewards received at the next timesteps.
        next_obses : torch.Tensor
            The observations at the next timesteps.
        not_dones : torch.Tensor
            Whether the episodes terminated at the next timesteps.
        not_dones_no_max : torch.Tensor
            Whether the episodes terminated at the next timesteps, disregarding
            the `max_episode_steps` limit.
        """

        # Get random indices corresponding to the batch size
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)  

        # Convert the sampled observations to tensors and move to device (cuda)
        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(self.next_obses[idxs],
                                     device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs],
                                           device=self.device) 

        # Return the sampled transitions
        return obses, actions, rewards, next_obses, not_dones, not_dones_no_max 
    
    
    # TODO: Implement sample state entropy