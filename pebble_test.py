# ------------------------------- 1. IMPORTING USEFUL LIBRARIES -----------------------------
#!/usr/bin/env python3
from hashlib import file_digest
import numpy as np  # ✅
import torch # type: ignore #✅
import torch.nn as nn # type: ignore # ✅
import torch.nn.functional as F # type: ignore #✅

import copy # object copy  ✅
import math # basic math functions ✅
import os # file paths ✅
import sys # system operations  ✅
import time # time operations ✅
import pickle as pkl # serialisation ✅
import tqdm # type: ignore # progress bar ✅

from logger import Logger  # type: ignore # custom logger ✅
from replay_buffer import ReplayBuffer # type: ignore # custom replay buffer ✅
from reward_model import RewardModel # custom reward model ✅
from collections import deque # double ended queue ✅
from obs_process import obs_process, get_process_obs_dim  # type: ignore # observation processing ❌

import utils # type: ignore # utility functions  such as set_seed_everywhere, make_env, get_process_obs_dim, and obs_process ❌
import hydra # type: ignore # hyperparameter management ✅


# os.environ["MUJOCO_GL"] = "osmesa" # configures the rendering backend for MuJoCo environments, ensuring compatibility when using off-screen rendering  ✅


# ------------------------------- 2. WORKSPACE CLASS -----------------------------------
#  sets up the environment, agent, replay buffer, reward model, and manages training, evaluation, and checkpointing. 
class Workspace(object):
    
    def __init__(self, cfg): # ✅
        
        # --------------------- 3. Configuration and env setup -------------------------
        # Store and print the current working directory
        self.work_dir = os.getcwd() # ✅
        print(f'workspace: {self.work_dir}') # ✅
        # store the configuration info in self.cfg
        self.cfg = cfg # ✅
        # A custom logger is instantiated to record training metrics, with options for TensorBoard logging and frequency control.
        self.logger = Logger( # ✅
            self.work_dir, # ✅
            save_tb=cfg.log_save_tb, # ✅
            log_frequency=cfg.log_frequency, # ✅
            agent=cfg.agent.name, # ✅
            )
        # A global seed is set via utils.set_seed_everywhere to ensure reproducibility.
        utils.set_seed_everywhere(cfg.seed) # ✅
        # The device is set based in the cfg file
        self.device = torch.device(cfg.device) # ✅
        self.log_success = False # ✅
        # The environment is created using utils.make_env and stored in self.env
        self.env = utils.make_env(self.cfg) # ✅
        # The initial observation is processed (using obs_process) to determine the shape needed by the agent and reward model.
        obs = self.env.reset() # ✅
        process_obs_shape = get_process_obs_dim(self.cfg.env, obs, self.cfg.process_type) # ✅
        # The observation dimension (obs_dim) and action space dimensions are stored in the agent’s configuration. The action range is also set.
        self.cfg.agent.params.obs_dim = process_obs_shape[0] # ✅
        self.cfg.agent.params.action_dim = self.env.action_space.shape[0] # ✅
        self.cfg.agent.params.action_range = [ # ✅
            float(self.env.action_space.low.min()),  # ✅
            float(self.env.action_space.high.max()) # ✅
        ]
        # The agent is created via Hydra’s instantiation utility. This allows the actual network and algorithm to be defined in the configuration.
        self.agent = hydra.utils.instantiate(self.cfg.agent) # ✅
        # A ReplayBuffer is created to store experience tuples. Its size is determined by the configuration, and it is set to work on the correct device.
        self.replay_buffer = ReplayBuffer( # ✅
            process_obs_shape, # ✅
            self.env.action_space.shape, # ✅
            int(self.cfg.replay_buffer_capacity), # ✅
            self.cfg.traj_action, # ✅
            self.device) # ✅
        # set path for saving trajectories
        traj_save_path = f"{self.work_dir}/{self.cfg.traj_save_path}" if self.cfg.traj_save_path is not None else None # ✅
        # A RewardModel is instantiated with numerous hyperparameters
        self.reward_model = RewardModel( # ✅
            process_obs_shape[0], # ✅
            # obs_dim_for_reward,
            self.env.action_space.shape[0], # ✅
            ensemble_size=self.cfg.ensemble_size, # ✅
            size_segment=self.cfg.segment, # ✅
            activation=self.cfg.activation,  # ✅
            lr=self.cfg.reward_lr, # ✅
            mb_size=self.cfg.reward_batch,  # ✅
            large_batch=self.cfg.large_batch, # ❌
            label_margin=self.cfg.label_margin,  # ✅
            teacher_beta=self.cfg.teacher_beta,  # ✅
            teacher_gamma=self.cfg.teacher_gamma, # ✅
            teacher_eps_mistake=self.cfg.teacher_eps_mistake,  # ✅
            teacher_eps_skip=self.cfg.teacher_eps_skip, # ✅
            teacher_eps_equal=self.cfg.teacher_eps_equal, # ✅
            env_name=self.cfg.env, # ✅
            traj_action=self.cfg.traj_action, # ✅
            traj_save_path=traj_save_path, # ✅
            vlm_label=self.cfg.vlm_label, # ✅
            better_traj_gen=self.cfg.better_traj_gen, # ✅
            double_check=self.cfg.double_check, # ✅
            save_equal=self.cfg.save_equal, # ✅
            vlm_feedback=self.cfg.vlm_feedback, # ✅
            generate_check=self.cfg.generate_check) # ✅
        # Several counters and accuracy metrics for queries and labels are initialised
        self.step = 0 # ✅
        self.total_feedback = 0 # ✅
        self.labeled_feedback = 0 # ✅
        self.llm_query_accuracy = 0 # ✅
        self.llm_label_accuracy = 0 # ✅

    # --------------------- 4. Saving and Loading Checkpoints ---------------------
    # Saves the current state of training including step count, replay buffer, and feedback statistics. The method creates a dictionary capturing the training state. It also uses Python’s pickle to dump this checkpoint to a file. Additionally, it calls the save methods on both the agent and the reward model so that their weights and configurations are persisted. Finally, A message is printed indicating the checkpoint was saved.
    def save_checkpoint(self): # ✅
        checkpoint = {
            'step': self.step, # ✅
            'replay_buffer': self.replay_buffer, # ✅
            'total_feedback': self.total_feedback, # ✅
            'labeled_feedback': self.labeled_feedback, # ✅
            'llm_query_accuracy': self.llm_query_accuracy, # ✅
            'llm_label_accuracy': self.llm_label_accuracy # ✅
        }
        with open(self.checkpoint_path, 'wb') as f: # ✅
            pkl.dump(checkpoint, f) # ✅
        print(f'Checkpoint saved at step {self.step}') # ✅
        # Save agent and reward model separately
        self.agent.save(self.work_dir, 'latest') # ✅
        self.reward_model.save(self.work_dir, 'latest') # ✅
    # Resumes training from a saved checkpoint. The checkpoint file is loaded and the training state (step count, replay buffer, and feedback statistics) is restored. The agent and reward model states are also loaded via their own load methods. A message is printed to confirm resumption from the checkpoint.
    def load_checkpoint(self): # ✅
        with open(self.checkpoint_path, 'rb') as f: # ✅
            checkpoint = pkl.load(f) # ✅
        self.step = checkpoint['step'] # ✅
        self.replay_buffer = checkpoint['replay_buffer'] # ✅
        self.total_feedback = checkpoint['total_feedback'] # ✅
        self.labeled_feedback = checkpoint['labeled_feedback'] # ✅
        self.llm_query_accuracy = checkpoint['llm_query_accuracy'] # ✅
        self.llm_label_accuracy = checkpoint['llm_label_accuracy'] # ✅
        print(f'Resuming from checkpoint at step {self.step}') # ✅
        # Load agent and reward model separately
        self.agent.load(self.work_dir, 'latest') # ✅
        self.reward_model.load(self.work_dir, 'latest') # ✅


    # The evaluate function is responsible for testing the agent’s performance over a set number of episodes and logging key metrics
    def evaluate(self):  # ✅ - PASS 
        # to accumulate the total rewards and success information over all evaluation episodes.
        # Often in reinforcement learning projects, there can be a distinction between a predicted or processed reward (which might be used internally for training) and the “true” reward given by the environment. This function tracks both.
        average_episode_reward = 0 # ✅
        average_true_episode_reward = 0 # ✅
        success_rate = 0 # ✅
        # Creates a directory within the logger’s directory to save evaluation videos (GIFs) if video recording is enabled. It combines the log directory path with a subdirectory name (eval_gifs). If that directory doesn’t exist, it is created with os.makedirs.
        save_gif_dir = os.path.join(self.logger._log_dir, 'eval_gifs') # ✅
        if not os.path.exists(save_gif_dir): # ✅
            os.makedirs(save_gif_dir) # ✅
        # for each episode  
        for episode in range(self.cfg.num_eval_episodes): # ✅
            images = [] # ✅
            obs = self.env.reset() # ✅
            # The raw observation is passed through a function (obs_process) that standardizes or transforms it based on the environment and the chosen processing type. The agent is reset to clear any episodic state (like recurrent hidden states) before evaluation.
            obs = obs_process(self.cfg.env, obs, self.cfg.process_type) # ✅
            self.agent.reset() # ✅
            done = False # determines when the episode has ended. # ✅
            # episode_reward and true_episode_reward are initialized to zero for accumulating rewards throughout the episode.
            episode_reward = 0 # ✅
            true_episode_reward = 0 # ✅
            # If the environment supports a “success” metric (e.g., for tasks where achieving a goal is binary), episode_success is initialized.
            if self.log_success: # ✅
                episode_success = 0 # ✅

            while not done:
                # The agent is put into evaluation mode (often disabling dropout or exploration noise) using a context manager. The agent then selects an action deterministically (sample=False). 
                with utils.eval_mode(self.agent): # ✅
                    action = self.agent.act(obs, sample=False) # ✅
                # Executes the chosen action in the environment. 
                try: 
                    obs, reward, done, extra = self.env.step(action) # ✅
                except:
                    obs, reward, terminated, truncated, extra = self.env.step(action) # ✅
                    done = terminated or truncated # ✅
                # The new observation is processed just like the initial one. Both the processed and “true” rewards are accumulated. If the environment reports a success flag (found in the extra dictionary), episode_success is updated with the maximum value observed during the episode (e.g., if success is signaled at any time, it’s recorded).
                obs = obs_process(self.cfg.env, obs, self.cfg.process_type) # ✅
                episode_reward += reward # ✅
                true_episode_reward += reward # ✅
                if self.log_success: # ✅
                    episode_success = max(episode_success, extra['success']) # ✅
            # After the while loop for an episode finishes, The episode’s rewards and success are added to the running totals.
            average_episode_reward += episode_reward # ✅
            average_true_episode_reward += true_episode_reward # ✅
            if self.log_success: # ✅
                success_rate += episode_success # ✅
            # If video saving is enabled, a file name is generated that includes the current training step, episode number, and rounded reward.
            if self.cfg.save_video: # ✅
                save_gif_path = os.path.join(save_gif_dir, 'step{:07}_episode{:02}_{}.gif'.format(self.step, episode, round(true_episode_reward, 2))) # ✅
                utils.save_numpy_as_gif(np.array(images), save_gif_path) # ✅
        # After processing all evaluation episodes, the average reward and success rate are calculated.
        average_episode_reward /= self.cfg.num_eval_episodes # ✅
        average_true_episode_reward /= self.cfg.num_eval_episodes # ✅
        if self.log_success: # ✅
            success_rate /= self.cfg.num_eval_episodes # ✅
            success_rate *= 100.0 # ✅
        # The metrics are logged with appropriate tags along with the current training step.
        self.logger.log('eval/episode_reward', average_episode_reward,
                        self.step) # ✅
        self.logger.log('eval/true_episode_reward', average_true_episode_reward,
                        self.step) # ✅
        if self.log_success:
            self.logger.log('eval/success_rate', success_rate,
                    self.step) # ✅
        self.logger.dump(self.step) # ✅
        # Closes the environment to free up resources and ensure a clean exit from the evaluation process.
        self.env.close() # ✅
    
    
    def evaluate_reward_model(self):
        # TODO: Call get train accuracy to get the accuracy of each ensemble member. Log to a log file
        pass

    # The purpose of learn_reward is to update the reward model based on feedback sampled from the replay buffer or collected data. It first selects which queries to label (i.e., which experiences to use for training the reward model) using various sampling strategies. Then, it trains the reward model for a set number of epochs until the training accuracy reaches a desired threshold.
    def learn_reward(self, first_flag=0):            
        # get feedbacks - Initializes counters for the number of queries that are labeled (i.e., get clean feedback) and those that might be noisy
        labeled_queries, noisy_queries = 0, 0
        # If first_flag is set to 1, it indicates that this is the initial feedback round. The function then uses uniform sampling to randomly select queries from the data. This can help in ensuring a diverse initial set of labeled examples.
        if first_flag == 1:
            # if it is first time to get feedback, need to use random sampling
            labeled_queries = self.reward_model.uniform_sampling()
        # For subsequent feedback rounds (when first_flag is not 1), the sampling strategy is chosen based on the configuration (self.cfg.feed_type):
        # 0: Uniform Sampling – Randomly select queries.
        # 1: Disagreement Sampling – Select queries where ensemble members disagree the most, which might indicate uncertain areas.
        # 2: Entropy Sampling – Choose queries with the highest entropy (uncertainty) in the reward predictions.
        # 3: k-Center Sampling – Sample queries that best cover the data space.
        # 4: k-Center Disagreement Sampling – A combination of k-center and disagreement sampling.
        # 5: k-Center Entropy Sampling – A combination of k-center and entropy sampling.
        else:
            if self.cfg.feed_type == 0:
                labeled_queries = self.reward_model.uniform_sampling()
            elif self.cfg.feed_type == 1:
                labeled_queries = self.reward_model.disagreement_sampling()
            elif self.cfg.feed_type == 2:
                labeled_queries = self.reward_model.entropy_sampling()
            elif self.cfg.feed_type == 3:
                labeled_queries = self.reward_model.kcenter_sampling()
            elif self.cfg.feed_type == 4:
                labeled_queries = self.reward_model.kcenter_disagree_sampling()
            elif self.cfg.feed_type == 5:
                labeled_queries = self.reward_model.kcenter_entropy_sampling()
            else:
                raise NotImplementedError
        # self.total_feedback is incremented by the mini-batch size (mb_size) of the reward model. This likely represents the total number of feedback samples that were requested in this round.
        # self.labeled_feedback is increased by the number of queries that received labels (i.e., those selected by the sampling strategy).
        self.total_feedback += self.reward_model.mb_size
        self.labeled_feedback += labeled_queries
        # If there are any labeled feedback samples, the reward model will be trained. The training runs for a maximum number of epochs defined by self.cfg.reward_update.
        train_acc = 0
        if self.labeled_feedback > 0:
            # update reward
            for epoch in range(self.cfg.reward_update):
                # additional soft constraints are being used
                if self.cfg.label_margin > 0 or self.cfg.teacher_eps_equal > 0 or ((self.cfg.vlm_label is not None) and (self.cfg.save_equal)):
                    self.llm_query_accuracy, llm_label_accuracy, train_acc = self.reward_model.train_soft_reward()
                else:
                    # This is the standard training routine used when the above conditions are not met.
                    # llm_query_accuracy: Accuracy of the queries as evaluated by the model or teacher.
                    # llm_label_accuracy: Accuracy of the labels provided.
                    # train_acc: The training accuracy or performance metric of the reward model for that epoch.
                    self.llm_query_accuracy, llm_label_accuracy, train_acc = self.reward_model.train_reward()
                # The training accuracy (train_acc) is averaged to produce total_acc.
                # The label accuracy is also averaged.
                # If the training accuracy exceeds a threshold of 97%, the training loop breaks early. This early stopping criterion prevents over-training once the reward model is performing well on the labeled data.
                total_acc = np.mean(train_acc)
                self.llm_label_accuracy = np.mean(llm_label_accuracy)
                if total_acc > 0.97:
                    break  
        # After training the reward model, the function prints the final training accuracy and label accuracy to provide feedback on the quality of the reward function update.     
        print("Reward function is updated!! ACC: " + str(total_acc))
        print("Reward function is updated!! LLM LABEL ACC: " + str(self.llm_label_accuracy))


    def run(self):
        # episode - Counts how many episodes have been run.
        # episode_reward - Accumulates the total reward for the current episode.
        # true_episode_reward - Accumulates the true reward given by the environment.
        episode, episode_reward, done = 0, 0, True
        # If the environment provides a success signal (e.g., for goal-based tasks), it is tracked here. 
        if self.log_success:
            episode_success = 0
        true_episode_reward = 0
        
        # store train returns of recent 10 episodes - This running average is later used to update margins and scheduling parameters.
        avg_train_true_return = deque([], maxlen=10) 
        # records when an episode begins, used later to log the episode duration.
        start_time = time.time()
        # counts how many environment interactions (steps) have occurred since the last reward model update.
        interact_count = 0
        first_evaluate = 0  # prepared to track evaluation status
        
        # ------------------------ Main training loop ------------------------
        # The loop runs until the total number of training steps reaches a preconfigured limit (num_train_steps).
        while self.step < self.cfg.num_train_steps:
            
            # if the episode has ended
            if done:
                # If not at the very start, the duration of the episode is logged:
                if self.step > 0:
                    self.logger.log('train/duration', time.time() - start_time, self.step)
                    start_time = time.time()   # This is done to capture the episode's duration
                    self.logger.dump(
                        self.step, save=(self.step > self.cfg.num_seed_steps))
                    
                # evaluate agent periodically
                if self.step > 0 and self.step % self.cfg.eval_frequency == 0:
                    self.logger.log('eval/episode', episode, self.step)
                    self.evaluate()
                
                # log important metrics
                self.logger.log('train/episode_reward', episode_reward, self.step)
                self.logger.log('train/true_episode_reward', true_episode_reward, self.step)
                self.logger.log('train/total_feedback', self.total_feedback, self.step)
                self.logger.log('train/labeled_feedback', self.labeled_feedback, self.step)
                self.logger.log('train/llm_query_accuracy', self.llm_query_accuracy, self.step)
                self.logger.log('train/llm_label_accuracy', self.llm_label_accuracy, self.step)
                if self.log_success:
                    self.logger.log('train/episode_success', episode_success,
                        self.step)
                    self.logger.log('train/true_episode_success', episode_success,
                        self.step)
                
                # reset env and agent
                obs = self.env.reset()
                obs = obs_process(self.cfg.env, obs, self.cfg.process_type)
                self.agent.reset()
                done = False
                episode_reward = 0
                avg_train_true_return.append(true_episode_reward)
                true_episode_reward = 0
                if self.log_success:
                    episode_success = 0
                episode_step = 0
                episode += 1
                self.logger.log('train/episode', episode, self.step)
            
            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample() # random sampling
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True) # policy sampling

            # -------------------------- Model update phase --------------------------
            # At the end of the unsupervised exploration phase
            if self.step == (self.cfg.num_seed_steps + self.cfg.num_unsup_steps):
                # update schedule - A fraction (frac) is computed based on how many training steps remain relative to the total training steps. This fraction is used to adjust the reward model’s mini-batch size or learning schedule.
                if self.cfg.reward_schedule == 1:
                    frac = (self.cfg.num_train_steps-self.step) / self.cfg.num_train_steps
                    if frac == 0:
                        frac = 0.01
                elif self.cfg.reward_schedule == 2:
                    frac = self.cfg.num_train_steps / (self.cfg.num_train_steps-self.step +1)
                else:
                    frac = 1
                self.reward_model.change_batch(frac)
                
                # update margin --> not necessary / will be updated soon
                # To determine whether to skip a label or not. This is to avoid making too many uodates to the reward model on little/insignificant gains. If the difference is smaller than the margin, the labels are considered equal and not used. If large, the label is used. 
                new_margin = np.mean(avg_train_true_return) * (self.cfg.segment / self.env._max_episode_steps)
                self.reward_model.set_teacher_thres_skip(new_margin)
                self.reward_model.set_teacher_thres_equal(new_margin)
                
                # first learn reward - first flag of 1 -> use uniform sampling strategy for the update
                self.learn_reward(first_flag=1)
                # relabel buffer
                self.replay_buffer.relabel_with_predictor(self.reward_model)
                # reset Q due to unsuperivsed exploration
                self.agent.reset_critic()
                # update agent
                self.agent.update_after_reset(
                    self.replay_buffer, self.logger, self.step, 
                    gradient_update=self.cfg.reset_update, 
                    policy_update=True)
                # reset interact_count - The counter tracking the number of interactions since the last reward update is reset to zero.
                interact_count = 0
                
            # ***************************************************************   
            #  Enough data has been collected  
            elif self.step > self.cfg.num_seed_steps + self.cfg.num_unsup_steps:
                # update reward function
                # Ensure that the accumulated teacher feedback has not exceeded a maximum limit. This budget prevents overuse of expensive or limited teacher feedback.
                if self.total_feedback < self.cfg.max_feedback:
                    # Only trigger the reward model update after a fixed number of interactions (e.g., every 5,000 steps). This batch-update approach lets the system gather a chunk of new data before updating.
                    if interact_count == self.cfg.num_interact:
                        # update schedule
                        if self.cfg.reward_schedule == 1:
                            frac = (self.cfg.num_train_steps-self.step) / self.cfg.num_train_steps
                            if frac == 0:
                                frac = 0.01
                        elif self.cfg.reward_schedule == 2:
                            frac = self.cfg.num_train_steps / (self.cfg.num_train_steps-self.step +1)
                        else:
                            frac = 1
                        self.reward_model.change_batch(frac)
                        
                        # update margin --> not necessary / will be updated soon
                        new_margin = np.mean(avg_train_true_return) * (self.cfg.segment / self.env._max_episode_steps)
                        self.reward_model.set_teacher_thres_skip(new_margin * self.cfg.teacher_eps_skip)
                        self.reward_model.set_teacher_thres_equal(new_margin * self.cfg.teacher_eps_equal)
                        
                        # corner case: new total feed > max feed
                        if self.reward_model.mb_size + self.total_feedback > self.cfg.max_feedback:
                            self.reward_model.set_batch(self.cfg.max_feedback - self.total_feedback)
                            
                        self.learn_reward()
                        self.replay_buffer.relabel_with_predictor(self.reward_model)
                        interact_count = 0
                        
                self.agent.update(self.replay_buffer, self.logger, self.step, 1)
                
            # unsupervised exploration
            elif self.step > self.cfg.num_seed_steps:
                self.agent.update_state_ent(self.replay_buffer, self.logger, self.step, 
                                            gradient_update=1, K=self.cfg.topK)
                
            try:
                next_obs, reward, done, extra = self.env.step(action)
            except:
                next_obs, reward, terminated, truncated, extra = self.env.step(action)
                done = terminated or truncated

            next_obs = obs_process(self.cfg.env, next_obs, self.cfg.process_type)

            if self.cfg.traj_action:
                reward_hat = self.reward_model.r_hat(np.concatenate([obs, action], axis=-1))
            else:
                reward_hat = self.reward_model.r_hat(obs)
                # obs_reward = np.concatenate([obs[0:2], obs[4:6]], axis=-1)
                # reward_hat = self.reward_model.r_hat(obs_reward)

            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward_hat
            true_episode_reward += reward
            
            if self.log_success:
                episode_success = max(episode_success, extra['success'])
                
            # adding data to the reward training data
            self.reward_model.add_data(obs, action, reward, done)
            # self.reward_model.add_data(obs_reward, action, reward, done)
            self.replay_buffer.add(
                obs, action, reward_hat, 
                next_obs, done, done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1
            interact_count += 1

            if self.step % 10000 == 0:
                self.save_checkpoint()
        
@hydra.main(config_path='config/train_PEBBLE.yaml', strict=True)
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()
  
if __name__ == '__main__':
    main()