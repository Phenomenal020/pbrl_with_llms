from stable_baselines3.dqn.dqn import DQN

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

class CustomDQN(DQN):
    """
    Custom Deep Q-Network (DQN)
    """

    # ---------------------------------✅---------------------------------
    def __init__(self, *args, reward_model=None, replay_buffer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_model = reward_model
        self.replay_buffer = replay_buffer

    # --------------------------------------------------------------------

    def train(self, gradient_steps: int, batch_size: int = 100, reward_model=None) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)


        # -------------------------------✅-----------------------------------
       # Train the reward model until the average ensemble accuracy is at least 0.95.
        while True:
            # Uniformly sample queries to obtain labels and update the preferences buffer.
            num_labels = self.reward_model.uniform_sampling()
            if num_labels > 0:
                llm_query_accuracy, llm_label_accuracy, ensemble_acc = self.reward_model.train_reward()
            else:
                print("No valid labels available for reward model training.")

            # Compute the average ensemble accuracy.
            avg_acc = np.mean(ensemble_acc)
            print(f"Average Ensemble Accuracy: {avg_acc:.4f}")

            if avg_acc >= 0.95:
                print("Reward model training complete.")
                break
            else:
                print(f"Continuing training... Accuracy is {avg_acc:.4f}")

        # After training, use the trained reward model to update the rewards in the replay buffer.
        self.replay_buffer.relabel_with_predictor(self.reward_model)

        # --------------------------------------------------------------------

        print("Model training started.")
        
        losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            with th.no_grad():
                # Compute the next Q-values using the target network
                next_q_values = self.q_net_target(replay_data.next_observations)
                # Follow greedy policy: use the one with the highest value
                next_q_values, _ = next_q_values.max(dim=1)
                # Avoid potential broadcast issue
                next_q_values = next_q_values.reshape(-1, 1)
                # 1-step TD target
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates
            current_q_values = self.q_net(replay_data.observations)

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

            # Compute Huber loss (less sensitive to outliers)
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        print("Model training completed.")

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
        # ----------------------------✅--------------------------------------
        # Log LLM query accuracy and LLM label accuracy.
        self.logger.record("train/llm_query_accuracy", self.reward_model.llm_query_accuracy)
        self.logger.record("train/llm_label_accuracy", self.reward_model.llm_label_accuracy)
        # --------------------------------------------------------------------