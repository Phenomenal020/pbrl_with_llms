# Source: https://stable-baselines.readthedocs.io/en/master/guide/pretrain.html#pre-train-a-model-using-behavior-cloning

import argparse
import numpy as np
from imitation.data.types import Transitions
from pyrcareworld.envs.bathing_env import BathingEnv
from naive_rl import ReachSponge

from imitation.data.types import Trajectory, Transitions
from imitation.data.rollout import flatten_trajectories

from imitation.algorithms import bc
from stable_baselines3.common.evaluation import evaluate_policy

from naive_rl import ReachSponge

from imitation.data import rollout


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
        # Each episode's obs should be an array of shape (episode_len + 1, observation_shape)
        # and acts of shape (episode_len, action_shape)
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
    print("Observations shape:", transitions.obs.shape)
    print("Actions shape:", transitions.acts.shape)
    print("Next Observations shape:", transitions.next_obs.shape)
    print("Dones shape:", transitions.dones.shape)

    rng = np.random.default_rng()
    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=transitions,
        rng=rng,
    )
    
    reward_before_training, _ = evaluate_policy(bc_trainer.policy, env, 40)
    print(f"Reward before training: {reward_before_training}")
    
    bc_trainer.train(n_epochs=20, reset_tensorboard=False)
    reward_after_training, _ = evaluate_policy(bc_trainer.policy, env, 40)
    print(f"Reward after training: {reward_after_training}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run RCareWorld bathing environment simulation.')
    parser.add_argument('-g', '--graphics', action='store_true', help='Enable graphics')
    parser.add_argument('-d', '--dev', action='store_true', help='Run in developer mode')
    args = parser.parse_args()
    _main(use_graphics=args.graphics, dev=args.dev)





# Run tensorboard: tensorboard --logdir logs/pretrain/bc
# Then open your web browser and navigate to http://localhost:6006 (by default) to see the training progress.