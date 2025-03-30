# Source: 
# https://imitation.readthedocs.io/en/latest/tutorials/3_train_gail.html

import argparse

from naive_rl import ReachSponge

from imitation.data.types import Trajectory
from imitation.data.rollout import flatten_trajectories

import numpy as np

from imitation.algorithms.adversarial.gail import GAIL
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from logger import pretrain_logger

SEED = 42


def _main(use_graphics=False, dev=None):
    # Load data from the npz file.
    try:
        data = np.load("expert_reach_sponge.npz", allow_pickle=True)
    except FileNotFoundError:
        print("Expert data file not found. Please run behaviour_cloning.py first.")
        pretrain_logger.error("Expert data file not found. Please run behaviour_cloning.py first.")
        return

    # These are stored as lists/arrays of episodes.
    episodes_obs = data["obs"]
    episodes_acts = data["acts"]
    episodes_dones = data["dones"]

    trajectories = []
    for i in range(len(episodes_obs)):
        # Each episode's obs should be an array of shape (episode_len + 1, observation_shape) and acts of shape (episode_len, action_shape)
        traj_obs = np.array(episodes_obs[i])
        traj_acts = np.array(episodes_acts[i])
        episodes_info = data["info"]
        
        # For info, we use the full info list for the episode.
        traj_infos = np.array(episodes_info[i]) if isinstance(episodes_info[i], (list, np.ndarray)) else episodes_info[i]
        
        # The trajectory's terminal flag is taken from the last "done" of the episode.
        terminal = bool(episodes_dones[i][-1])
        
        trajectory = Trajectory(obs=traj_obs,
                                acts=traj_acts,
                                # next_obs=traj_obs[1:],
                                infos=traj_infos,
                                terminal=terminal)
        trajectories.append(trajectory)

    # Now flatten the trajectories to get a continuous Transitions object.
    transitions = flatten_trajectories(trajectories)

    # Print/verify
    print("Transitions object created:")
    print("Observations shape:", type(transitions.obs))
    print("Actions shape:", type(transitions.acts))
    print("Next Observations shape:", transitions.next_obs)
    print("Dones shape:", transitions.dones)
    pretrain_logger.gail_logger.info("Transitions object created:")
    pretrain_logger.gail_logger.info("| Metric            | Value               |")
    pretrain_logger.gail_logger.info("|-------------------|---------------------|")
    pretrain_logger.gail_logger.info(f"| Observations      | {transitions.obs} |")
    pretrain_logger.gail_logger.info(f"| Actions          | {transitions.acts} |")
    pretrain_logger.gail_logger.info(f"| Next Observations| {transitions.next_obs} |")
    pretrain_logger.gail_logger.info(f"| Dones            | {transitions.dones} |")   
    
    
    def make_env(port):
        env = ReachSponge(use_graphics=False, port=port)  # Pass the desired port
        env = Monitor(env, filename=f"models/pre_train/ppo/reach_sponge_monitor_{port}.csv")
        return env

    # Manually specify the ports you want to use for each environment
    manual_ports = [5005, 5261, 5517, 5750, 5801, 5823, 5837, 5851]

    # Create a list of lambda functions for each environment, each with its assigned port
    env_fns = [lambda port=port: make_env(port) for port in manual_ports]

    # Wrap the environments in a vectorized environment; you can use DummyVecEnv or SubprocVecEnv
    venv = SubprocVecEnv(env_fns)
    # rng = np.random.default_rng()
    
    learner = PPO(
        env=venv,
        policy=MlpPolicy,
        batch_size=128,
        ent_coef=0.0,
        learning_rate=0.0004,
        gamma=0.95,
        n_epochs=8, 
    )
    reward_net = BasicRewardNet(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        normalize_input_layer=RunningNorm,
    )
    gail_trainer = GAIL(
        demonstrations=transitions,
        demo_batch_size=32,
        gen_replay_buffer_capacity=512,
        n_disc_updates_per_round=8,
        venv=venv,
        gen_algo=learner,
        reward_net=reward_net,
        allow_variable_horizon=True
    )
    
    # Verify your action space contains {0,1,2}
    print(venv.action_space.contains(0))
    print(venv.action_space.contains(1))
    print(venv.action_space.contains(2))
    
    # Before training
    # env.seed(SEED)
    learner_rewards_before_training, _ = evaluate_policy(
        learner, venv, 15, return_episode_rewards=True
    )
    print("Rewards before training:",
    np.mean(learner_rewards_before_training),
    "+/-",
    np.std(learner_rewards_before_training),
    )
    pretrain_logger.gail_logger.info(f"Rewards before training: {np.mean(learner_rewards_before_training)} +/- {np.std(learner_rewards_before_training)}")
    
    # Train
    gail_trainer.train(10000)
    pretrain_logger.gail_logger.info("Training completed.")
    print("Training completed.")
    
    # After training
    learner_rewards_after_training, _ = evaluate_policy(
        learner, venv, 30, return_episode_rewards=True
    )
    print("Rewards after training:",
        np.mean(learner_rewards_after_training),
        "+/-",
        np.std(learner_rewards_after_training),
    )
    pretrain_logger.gail_logger.info(f"Rewards after training: {np.mean(learner_rewards_after_training)} +/- {np.std(learner_rewards_after_training)}")
    
    # After training, save the PPO model:
    learner.save("models/pretrain/ppo_pretrained_model.zip")
    pretrain_logger.gail_logger.info("PPO model saved.")
    print("PPO model saved.")
    
    # Close the environment
    venv.close()
    pretrain_logger.gail_logger.info("Environment closed.")
    print("Environment closed.")

    pretrain_logger.gail_logger.info("Pretraining completed.")
    print("Pretraining completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run RCareWorld bathing environment simulation.')
    parser.add_argument('-g', '--graphics', action='store_true', help='Enable graphics')
    parser.add_argument('-d', '--dev', action='store_true', help='Run in developer mode')
    args = parser.parse_args()
    _main(use_graphics=args.graphics, dev=args.dev)