# Sources: 
# https://github.com/TU2021/RL-SaLLM-F/blob/main/prompt.py
# https://gymnasium.farama.org/environments/toy_text/frozen_lake/

from openai import OpenAI
client = OpenAI()


def traj_pair_process(obs1, obs2, acts1, acts2):
    # Check if the lengths of the trajectories are the same.
    if len(obs1) != len(obs2) or len(acts1) != len(acts2):
        raise ValueError("Trajectories must have the same length.")

    # Convert each component to a Python list if possible.
    traj1_data = {
        "agent_pos": obs1.tolist() if hasattr(obs1, "tolist") else obs1,
        "actions": acts1.tolist() if hasattr(acts1, "tolist") else acts1,
    }
    traj2_data = {
        "agent_pos": obs2.tolist() if hasattr(obs2, "tolist") else obs2,
        "actions": acts2.tolist() if hasattr(acts2, "tolist") else acts2,
    }

    return traj1_data, traj2_data


def stringify_trajs(traj1, traj2):
    # Use the custom numpy_encoder in json.dumps.
    traj1_str = json.dumps(traj1, indent=2, default=numpy_encoder)
    traj2_str = json.dumps(traj2, indent=2, default=numpy_encoder)

    return traj1_str, traj2_str


gpt_query_env_frozen_lake = """Suppose you are an expert robot trajectory evaluator. Now you need to evaluate the quality of the robot's motion trajectory. The environment is Frozen lake, which involves crossing a frozen lake from start to goal without falling into any holes. The player may not always move in the intended direction due to the slippery nature of the frozen lake. 

Description:
The game starts with the player at location 0 of the frozen lake grid world with the goal located at far extent of the world (15 for the 4x4 environment). The player makes moves until they reach the goal or fall in a hole. If they fall into a hole, they receive a reward of -5. If they reach the goal, they receive a reward of +15.

Action Space
The action shape is (1,) in the range {0, 3} indicating which direction to move the player.
0: Move left
1: Move down
2: Move right
3: Move up

Observation Space
The observation is a value representing the player’s current position as a value. 

Episode End
The episode ends if 
- The player moves into a hole (reward: -5)
- The player reaches the goal (reward: +15)

The following are two trajectories where:  
(1) "agent_pos" represents the position of the robot as a value;  
(2) "actions" represents the sequence of actions taken by the agent as defined in the action space.

Please answer the following two questions step by step:  
1. Is there any difference between Trajectory 1 and Trajectory 2 in terms of achieving the goal?  
Reply with your analysis of not more than 150 tokens.

2. Which trajectory do you think better achieves the goal?  
Reply a single line of 1 if you think the goal is better achieved in Trajectory 1, or 2 if it is better achieved in Trajectory 2. Reply 0 if the text is unsure or there is no significant difference.  
"""  


def gpt_infer(traj_str, traj_str2, prompt=gpt_query_env_frozen_lake):
    response = client.responses.create(
    model="o3-mini-2025-01-31",
    input=[
        {
        "role": "developer",
        "content": [
            {
            "type": "input_text",
            "text": f"{prompt}\n\nTrajectory 1:\n{traj_str}\n\nTrajectory 2:\n{traj_str2}\n\n"
            }
        ]
        },
    ],
    text={
        "format": {
        "type": "text"
        }
    },
    reasoning={
        "effort": "medium"
    },
    tools=[],
    store=True
    )

    return response.output_text