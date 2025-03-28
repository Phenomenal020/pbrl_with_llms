import os
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv
# from stable_baselines3 import DQN
from stable_baselines3.dqn.dqn import CustomDQN
from stable_baselines3.common.buffers import DQNPBRLReplayBuffer
from reward import RewardModel

# Ensure required directories exist.
os.makedirs("models", exist_ok=True)
os.makedirs("./frozen_lake/tensorboard_logs", exist_ok=True)
os.makedirs("./frozen_lake/traj", exist_ok=True)

class CustomFrozenLakeEnv(FrozenLakeEnv):
    def __init__(self, reward_model, **kwargs):
        super().__init__(**kwargs)
        self.reward_model = reward_model
        self.nsteps = 0
        self.nepisodes = 0
        self.eps_steps  = 0

    def step(self, action):
        # Call the original step method.
        obs, true_reward, terminated, truncated, info = super().step(action)
        # Calculate true reward in a sparse-reward environment.
        # (Note: Here true_reward is set to the observation value if the episode is ongoing; otherwise, -5.)
        true_reward = +obs if (not terminated and not truncated) else -5
        # Calculate predicted reward using the reward model.
        pred_reward = self.reward_model.r_hat(obs)
        # Add experience to the reward model's buffer.
        self.reward_model.add_data(
            obs, action, true_reward, pred_reward, terminated, truncated
        )
        self.nsteps += 1
        self.eps_steps += 1

        if self.eps_steps >= 24:
            truncated = True
            self.eps_steps = 0

        if terminated or truncated:
            self.nepisodes += 1
            
        return obs, pred_reward, terminated, truncated, info



class FrozenLakeTrainer:
    def __init__(self, total_timesteps=10000, save_interval=500, model_file="models/dqn_frozenlake.zip"):
        self.total_timesteps = total_timesteps
        self.save_interval = save_interval
        self.model_file = model_file
        self.tensorboard_log_dir = "./frozen_lake/tensorboard_logs"
        os.makedirs(self.tensorboard_log_dir, exist_ok=True)

        # Instantiate the reward model with the desired parameters.
        self.reward_model = RewardModel(
            ds=1,
            da=1,
            teacher_beta=1,
            teacher_gamma=0.99,
            teacher_eps_mistake=0,
            teacher_eps_skip=0,
            teacher_eps_equal=0,
            save_equal=True,
            traj_action=False,
            traj_save_path="./frozen_lake/traj",
        )
        # Create a replay buffer for a discrete observation and action space.
        self.replay_buffer = DQNPBRLReplayBuffer(
            buffer_size=10000,
            observation_space=gym.spaces.Discrete(16),
            action_space=gym.spaces.Discrete(4)
        )
        self.env = CustomFrozenLakeEnv(
            reward_model=self.reward_model,
            render_mode="human"
        )

        if os.path.exists(self.model_file):
            self.model = CustomDQN.load(self.model_file, env=self.env, tensorboard_log=self.tensorboard_log_dir)
            print("Model found. Resuming training from saved model.")
            self.reward_model.load()
        else:
            self.model = CustomDQN(
                learning_starts=400,
                env=self.env,
                policy="MlpPolicy", 
                reward_model=self.reward_model,
                replay_buffer=self.replay_buffer,
                tensorboard_log=self.tensorboard_log_dir, 
                train_freq=(10, "episode"), 
            )
            print("No model found. Starting afresh.")

    def train(self):
        timesteps_done = 0
        while timesteps_done < self.total_timesteps:
            steps_to_train = min(self.save_interval, self.total_timesteps - timesteps_done)
            self.model.learn(total_timesteps=steps_to_train, reset_num_timesteps=False, log_interval=4)
            timesteps_done += steps_to_train
            self.reward_model.save()
            self.model.save(self.model_file)
            print(f"Saved model at {timesteps_done} timesteps.")

    def evaluate(self):
        obs, info = self.env.reset()
        while True:
            action, _states = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.env.step(action)
            if terminated or truncated:
                obs, info = self.env.reset()

def main():
    trainer = FrozenLakeTrainer()
    trainer.train()
    trainer.evaluate()

if __name__ == "__main__":
    main()