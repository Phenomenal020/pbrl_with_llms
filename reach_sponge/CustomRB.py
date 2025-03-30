from stable_baselines3.common.buffers import ReplayBuffer

#  ---------------------------- PBRL CODE ----------------------------------
class DQNPBRLReplayBuffer(ReplayBuffer):
    
    # New method to relabel rewards with a reward model predictor
    def relabel_with_predictor(self, predictor, batch_size: int = 100):
        """
        Use the reward model predictor to relabel the rewards in the buffer in batches.

        :param predictor: A model with a method `r_hat_batch` that takes a batch of inputs and returns predicted rewards.
        :param batch_size: The number of samples per batch.
        """
        print("Relabeling rewards...")
        # Determine the total number of stored transitions.
        total_samples = self.buffer_size if self.full else self.pos
        
        # Batch processing
        total_iter = (total_samples + batch_size - 1) // batch_size  # Round up for any partial batch
        
        for index in range(total_iter):
            start_index = index * batch_size
            last_index = min((index + 1) * batch_size, total_samples)
            
            # Slice the batch of observations
            batch_obses = self.observations[start_index:last_index]
            
            # Get predicted rewards from the reward model in a batched fashion
            pred_rewards = predictor.r_hat_batch(batch_obses)
            
            # Replace the stored rewards
            self.rewards[start_index:last_index] = pred_rewards
        
        print("Relabeling rewards successfully...")
