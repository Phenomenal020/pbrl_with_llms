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


def _main(use_graphics=False, dev=None):
    
    env = ReachSponge()
    
    # log_dir = "logs/pretrain/bc"
    
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
    
    
    # reward_before_training, _ = evaluate_policy(bc_trainer.policy, env, 40)
    # print(f"Reward before training: {reward_before_training}")
    # print(bc_trainer.policy.load_state_dict())
    # print(bc_trainer.policy.state_dict())
    
    bc_trainer.train(n_epochs=10, reset_tensorboard=False)
    reward_after_training, _ = evaluate_policy(bc_trainer.policy, env, 40)
    print(f"Reward after training: {reward_after_training}")
    print(bc_trainer.policy.state_dict())
    # bc_trainer.policy.save("models/pretrain_dqn")
    
    
    # Reward after training: -36.16291978061199



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