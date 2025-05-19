# Source:
# https://imitation.readthedocs.io/en/latest/algorithms/bc.html


import argparse
import numpy as np

from pyrcareworld.envs.bathing_env import BathingEnv

from naive_rl import ReachSponge

from imitation.data.types import Trajectory, Transitions
from imitation.data.rollout import flatten_trajectories
from imitation.data import rollout
from imitation.algorithms import bc
from stable_baselines3.common.evaluation import evaluate_policy

from logger import pretrain_logger

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor


def _main(use_graphics=False, dev=None):
    
    env = ReachSponge()
    
    # Load data from the npz file.
    data = np.load("expert_reach_sponge.npz", allow_pickle=True)

    # Assume these are stored as lists or arrays of episodes.
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

    print("Transitions object created:")
    print("Observations shape:", transitions.obs)
    print("Actions shape:", transitions.acts)
    print("Next Observations shape:", transitions.next_obs)
    print("Dones shape:", transitions.dones)
    pretrain_logger.bc_logger.info("Transitions object created:")
    pretrain_logger.bc_logger.info("| Metric            | Value               |")
    pretrain_logger.bc_logger.info("|-------------------|---------------------|")
    pretrain_logger.bc_logger.info(f"| Observations      | {transitions.obs} |")
    pretrain_logger.bc_logger.info(f"| Actions          | {transitions.acts} |")
    pretrain_logger.bc_logger.info(f"| Next Observations| {transitions.next_obs} |")
    pretrain_logger.bc_logger.info(f"| Dones            | {transitions.dones} |")   

    rng = np.random.default_rng()
    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=transitions,
        rng=rng,
    )
    
    # Verify your action space contains {0,1,2}
    print(env.action_space.contains(0))
    print(env.action_space.contains(1))
    print(env.action_space.contains(2))
    
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
    
    
    reward_before_training, _ = evaluate_policy(bc_trainer.policy, venv, 16)
    print(f"Reward before training: {reward_before_training}")
    pretrain_logger.bc_logger.info(f"Reward before training: {reward_before_training}")
    
    bc_trainer.train(n_epochs=8, reset_tensorboard=False)
    reward_after_training, _ = evaluate_policy(bc_trainer.policy, venv, 16)
    
    print("Training completed.")
    pretrain_logger.bc_logger.info("Training completed.")
    
    print(f"Reward after training: {reward_after_training}")
    pretrain_logger.bc_logger.info(f"Reward after training: {reward_after_training}")
    
    # Save the policy
    bc_trainer.save_policy("models/pre_train/bc_policy")
    pretrain_logger.bc_logger.info("Policy saved.")    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run RCareWorld bathing environment simulation.')
    parser.add_argument('-g', '--graphics', action='store_true', help='Enable graphics')
    parser.add_argument('-d', '--dev', action='store_true', help='Run in developer mode')
    args = parser.parse_args()
    _main(use_graphics=args.graphics, dev=args.dev)





# Run tensorboard: tensorboard --logdir logs/pretrain/bc
# Then open your web browser and navigate to http://localhost:6006 (by default) to see the training progress.

# before train state dict

# <bound method Module.state_dict of FeedForward32Policy(
#   (features_extractor): FlattenExtractor(
#     (flatten): Flatten(start_dim=1, end_dim=-1)
#   )
#   (pi_features_extractor): FlattenExtractor(
#     (flatten): Flatten(start_dim=1, end_dim=-1)
#   )
#   (vf_features_extractor): FlattenExtractor(
#     (flatten): Flatten(start_dim=1, end_dim=-1)
#   )
#   (mlp_extractor): MlpExtractor(
#     (policy_net): Sequential(
#       (0): Linear(in_features=6, out_features=32, bias=True)
#       (1): Tanh()
#       (2): Linear(in_features=32, out_features=32, bias=True)
#       (3): Tanh()
#     )
#     (value_net): Sequential(
#       (0): Linear(in_features=6, out_features=32, bias=True)
#       (1): Tanh()
#       (2): Linear(in_features=32, out_features=32, bias=True)
#       (3): Tanh()
#     )
#   )
#   (action_net): Linear(in_features=32, out_features=3, bias=True)
#   (value_net): Linear(in_features=32, out_features=1, bias=True)
# )>