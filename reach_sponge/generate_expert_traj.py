# This code generates a .npz numpy archive file containing the expert data
# Later, the .npz file would be used for pre-training
# Load the .json file from expert_data.jsonl and save to a .npz file.
    
    
import json
import numpy as np

def _main():

    # initialise lists to hold the data 
    obs_l = []
    actions_l= []
    rewards_l = []
    episode_returns_l = []
    episode_starts_l= []
    dones_l = []
    info_l = []

    # Open and read the JSONL file
    with open('expert_traj.jsonl', 'r') as f:
        for line in f:
            if not line.strip():
                continue  # Skip any empty lines.
            data = json.loads(line) # Parse the json from the line
    #       #Convert lists back to numpy arrays
            obs_l.append(np.array(data["obs"]))
            actions_l.append(np.array(data["acts"]))
            rewards_l.append(np.array(data["rewards"]))
            episode_returns_l.append(np.array(data["episode_returns"]))
            episode_starts_l.append(np.array(data["episode_starts"]))
            dones_l.append(np.array(data["dones"]))
            info_l.append(np.array(data["info"]))       

   
    # # Save as a .npz file
    np.savez(
        "expert_reach_sponge.npz",
        obs=np.array(obs_l, dtype=object),
        acts=np.array(actions_l, dtype=object),
        # rewards=np.array(rewards_l, dtype=object),
        # episode_returns=np.array(episode_returns_l, dtype=object),
        # episode_starts=np.array(episode_starts_l, dtype=object)
        dones=np.array(dones_l, dtype=object),
        info=np.array(info_l, dtype=object)
    )
    
    print("File generated")
    
    
if __name__ == "__main__":
    _main()