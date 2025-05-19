#  Sources: 
# https://github.com/TU2021/RL-SaLLM-F/blob/main/prompt.py
# https://gymnasium.farama.org/environments/toy_text/frozen_lake/

import json
import re
import numpy as np

import jsonlines
import json
import os

from openai import OpenAI
client = OpenAI()

os.makedirs("./db/ablation/full_text", exist_ok=True)


def traj_pair_process(obs1, obs2, acts1, acts2, dones1, dones2, truncated1, truncated2, collision1, collision2, success1, success2):
    # Check if the lengths of the trajectories are the same.
    if len(obs1) != len(obs2) or len(acts1) != len(acts2) or len(dones1) != len(dones2) or len(truncated1) != len(truncated2) or len(collision1) != len(collision2) or len(success1) != len(success2):
        raise ValueError("Trajectories must have the same length.")

    # Convert each component to a Python list if possible.
    traj1_data = {
        "observations": obs1.tolist() if hasattr(obs1, "tolist") else obs1,
        "actions": acts1.tolist() if hasattr(acts1, "tolist") else acts1,
        "terminated": dones1.tolist() if hasattr(dones1, "tolist") else dones1,
        "truncated": truncated1.tolist() if hasattr(truncated1, "tolist") else truncated1,
        "collision": collision1.tolist() if hasattr(collision1, "tolist") else collision1,
        "success": success1.tolist() if hasattr(success1, "tolist") else success1
    }
    traj2_data = {
        "observations": obs2.tolist() if hasattr(obs2, "tolist") else obs2,
        "actions": acts2.tolist() if hasattr(acts2, "tolist") else acts2,
        "terminated": dones2.tolist() if hasattr(dones2, "tolist") else dones2,
        "truncated": truncated2.tolist() if hasattr(truncated2, "tolist") else truncated2,
        "collision": collision2.tolist() if hasattr(collision2, "tolist") else collision2,
        "success": success2.tolist() if hasattr(success2, "tolist") else success2
    }

    return traj1_data, traj2_data


def numpy_encoder(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def stringify_trajs(traj1, traj2, use_rag=False):
    if use_rag:
        expert_actions_1 = get_nearest_expert_demo(traj1["observations"][0])
        expert_actions_2 = get_nearest_expert_demo(traj2["observations"][0])
        print(f"expert_actions_1: {expert_actions_1}")
        print(f"expert_actions_2: {expert_actions_2}")
        expert_actions_1_str = json.dumps(expert_actions_1, indent=2, default=numpy_encoder)
        expert_actions_2_str = json.dumps(expert_actions_2, indent=2, default=numpy_encoder)
    # Use the custom numpy_encoder in json.dumps.
    traj1_str = json.dumps(traj1, indent=2, default=numpy_encoder)
    traj2_str = json.dumps(traj2, indent=2, default=numpy_encoder)
    if use_rag:
        return traj1_str, traj2_str, expert_actions_1_str, expert_actions_2_str
    return traj1_str, traj2_str


# ****************************************************************************



gpt_query_reach_sponge_no_rag = """Suppose you are an expert robot trajectory evaluator. You are provided with two trajectory segments from any two trajectories. Now you need to evaluate the quality of the robot's motion trajectory. The environment is a care bathing environment where a robot is expected to move towards a sponge inorder to grasp it. Consider priorities as (FIRST: success or collision (they are mutually exclusive); SECOND: progress towards the sponge; THIRD: truncation) in that order. BUT CONSIDER ALL and not exclusively one so your analysis is more robust.
Note: The segments may start at different locations, so progress should not only consider the distance to the sponge but also how much the robot has moved towards the sponge relative to its starting position in the segment.

Action Space:
- Discrete(3) with the following actions: 
0 -> Move forward by 0.2m
1 -> Turn left by 90 degrees, anti-clockwise
2 -> Turn right by 90 degrees, clockwise

Target/Goal:
Sponge Position -> [-0.10899999737739563, 0.5329999923706055, 2.2269999980926514]
Sponge Orientation -> [-0.0, 0.0, 0.0]

Observation Space:
- The observation is a numpy array representing the robot's [position, rotation] in the environment.

Done:
The episode ends (done = True): 
- Successfully if the robot moves to the sponge area - given by sponge_pos (highly desirable) OR
- The robot collides with an obstacle - collision (highly undesirable/detrimental) OR
- The episode is truncated.
Note: If it's the foremost option, the corresponding success value is true. Otherwise, it is false.

Truncated:
- The episode is truncated if the robot takes too many steps (n >= 100) or the robot moves away from the room - (discourage but not detrimental).

You are provided with two trajectory segments from a trajectory containing: 
(1) "observations" represents the pose of the robot as a numpy array; 
(2) "actions" represents the sequence of actions taken by the agent.
(3) "dones" represents whether the episode ended or not.
(4) "truncated" represents whether the episode was truncated or not - too many steps or the robot wandered too far from the room.
(5) "collision" represents whether the robot collides with an obstacle or not. ( 0 - No collision, 1+ - Collision)
(6) "success" represents whether the robot successfully reached the sponge or not.
Note: If no success nor collision nor done nor truncated, the trajectory is not completed and is ongoing.

Please answer the following questions step by step: 
1. Is there any difference between Trajectory 1 and Trajectory 2 in terms of achieving the goal? Which trajectory do you think better achieves the goal?  
2. Reply with a single number 1 if you think the goal is BETTER ACHIEVED in Trajectory 1, or 2 if it is BETTER ACHIEVED in Trajectory 2 and 0 if the text is unsure or there is no significant difference.
Format your response this way: [Final_Answer: 0 or 1 or 2]
"""

def gpt_infer_no_rag(traj_str, traj_str2, prompt=gpt_query_reach_sponge_no_rag, index=1):
    if index % 2 != 0:
        response = client.responses.create(
            model="o3-mini",
            input=[
                {
                    "role": "user",
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
            store=True,
        )
    else:
       response = client.responses.create(
            model="o3-mini",
            input=[
                {
                    "role": "user",
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
            store=True,
        )
        

    print(f"using model: {response.model} without rag")

    # Get the full output text from the response
    full_text = response.output_text
    input_token_count = response.usage.input_tokens
    output_token_count = response.usage.output_tokens

    with jsonlines.open("./db/ablation/full_text/full_text.jsonl", mode='a') as writer:
        writer.write({"full_text": full_text, "input_token_count": input_token_count, "output_token_count": output_token_count})
        print("Saved segment to ./db/ablation/full_text/full_text.jsonl") 

    # Extract the final answer using a regex that looks for "[Final_Answer: NUMBER]"
    match = re.search(r"\[Final_Answer:\s*([0-2])\]", full_text)
    if match:
        final_answer = match.group(1)
    else:
        # else, return the full text if the final answer is not found.
        final_answer = full_text.strip()

    print(f"Final Answer: {final_answer}")
    return final_answer





# ****************************************************************************

def get_nearest_expert_demo(starting_obs):
    """
    Given a starting observation, this function loads expert demonstrations
    from a JSONL file, finds the demonstration whose initial observation is the 
    most similar (using Euclidean distance), and returns the demonstration's actions.
    
    Parameters:
        starting_obs (array-like): The starting observation (should be a numeric vector).
        
    Returns:
        list: The actions from the demonstration with the closest starting observation,
              or None if no demonstration is found.
    """
    
    expert_demos = []
    try:
        # Load expert demonstrations from the JSONL file.
        with open("expert_traj.jsonl", "r") as f:
            # Process each line in the file.
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    demo = json.loads(line)
                    # Check that required keys exist
                    if "obs" in demo and "acts" in demo:
                        expert_demos.append(demo)
                    else:
                        print("Skipping demo without 'obs' or 'acts' keys")
                except json.JSONDecodeError as e:
                    print(f"Error decoding line: {line}. Error: {e}")
    except FileNotFoundError:
        print("Error: File expert_data.jsonl not found.")
        return None
    except Exception as e:
        print("Error loading expert demos:", e)
        return None
    
    if not expert_demos:
        print("No expert demonstrations loaded.")
        return None
    
    # Ensure starting_obs is a numpy array.
    starting_obs = np.array(starting_obs)
    print(f"Starting Observation: {starting_obs}")
    
    # Initialise variables to track the best match.
    min_dist = float('inf')
    best_actions_segment = None

    # Loop through each demonstration.
    for demo in expert_demos:
        # Verify that the demonstration has observations and actions.
        if not demo.get("obs") or not demo.get("acts"):
            continue
        
        obs_list = demo["obs"]
        actions_list = demo["acts"]

        # Loop through all observations in this demonstration.
        for idx, obs in enumerate(obs_list):
            # Convert the observation to a numpy array if it's not already.
            demo_obs = np.array(obs)
            # Compute the Euclidean distance between the starting observation and this observation.
            dist = np.linalg.norm(starting_obs - demo_obs)
            
            # Update the best match if this observation is closer.
            if dist < min_dist:
                min_dist = dist
                # Save the actions segment from this index to the end.
                best_actions_segment = actions_list[idx:]

    # Check whether a matching segment was found.
    if best_actions_segment is not None:
        return best_actions_segment
    else:
        # Handle the case when no valid demonstration was found.
        return []


gpt_query_reach_sponge_rag = """Suppose you are an expert robot trajectory evaluator. You are provided with two trajectory segments from any two trajectories AND expert-generated actions in a similar starting position. Now you need to evaluate the quality of the robot's motion trajectory. The environment is a care bathing environment where a robot is expected to move towards a sponge inorder to grasp it. Consider priorities as (FIRST: success or collision (they are mutually exclusive); SECOND: progress and how well it follows the expert's actions (assign equal weightage); THIRD: truncation) in that order. BUT CONSIDER ALL and not exclusively one so your analysis is more robust.
Note: The segments may start at different locations, so progress should not only consider the distance to the sponge but also how much the robot has moved towards the sponge relative to its starting position in the segment.

Action Space:
- Discrete(3) with the following actions: 
0 -> Move forward by 0.2m
1 -> Turn left by 90 degrees, anti-clockwise
2 -> Turn right by 90 degrees, clockwise

Target/Goal:
Sponge Position -> [-0.10899999737739563, 0.5329999923706055, 2.2269999980926514]
Sponge Orientation -> [-0.0, 0.0, 0.0]

Observation Space:
- The observation is a numpy array representing the robot's position and orientation in the environment.

Done:
The episode ends (done = True): 
- Successfully if the robot moves to the sponge area - given by sponge_pos (highly desirable) OR
- The robot collides with an obstacle - collision (highly undesirable/detrimental)
- The episode is truncated.
Note: If it's the foremost option, the corresponding success value is true. Otherwise, it is false.

Truncated:
- The episode is truncated if the robot takes too many steps (n >= 100) or the robot moves away from the room - (discourage but not detrimental).

You are provided with two trajectory segments from a trajectory containing: 
(1) "observations" represents the pose of the robot as a numpy array;  
(2) "actions" represents the sequence of actions taken by the robot.
(3) "dones" represents whether the episode ended or not.
(4) "truncated" represents whether the episode was truncated or not - too many steps or the robot wandered too far from the room.
(5) "collision" represents whether the robot collides with an obstacle or not. ( 0 - No collision, 1+ - Collision)
(6) "success" represents whether the robot successfully reached the sponge or not.
(7) "expert actions" represents the sequence of actions taken by an expert given a similar observation.
Note: If no success or collision nor done nor truncated, the trajectory is not completed and is ongoing.

Please answer the following questions step by step: 
1. Is there any difference between Trajectory 1 and Trajectory 2 in terms of achieving the goal? Which trajectory do you think better achieves the goal?  
2. Reply with a single number 1 if you think the goal is BETTER ACHIEVED in Trajectory 1, or 2 if it is BETTER ACHIEVED in Trajectory 2 and 0 if the text is unsure or there is no significant difference.
Format your response this way: [Final_Answer: 0 or 1 or 2]
"""

def gpt_infer_rag(traj_str, traj_str2, expert_actions_1_str, expert_actions_2_str, prompt=gpt_query_reach_sponge_rag, index=1):
    if index % 2 != 0:
       response = client.responses.create(
            model="o3-mini",
            input=[
                {
                    "role": "user",
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
            store=True,
        )
    else:
        response = client.responses.create(
            model="o3-mini",
            input=[
                {
                    "role": "user",
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
            store=True,
        )

    print(f"using model: {response.model} with rag")

    # Get the full output text from the response
    full_text = response.output_text
    input_token_count = response.usage.input_tokens
    output_token_count = response.usage.output_tokens

    print(f"Full Text: {full_text}")
    print(f"Input Token Count: {input_token_count}")
    print(f"Output Token Count: {output_token_count}")

    with jsonlines.open("./db/ablation/full_text/full_text.jsonl", mode='a') as writer:
        writer.write({"full_text": full_text, "input_token_count": input_token_count, "output_token_count": output_token_count})
        print("Saved segment to ./db/ablation/full_text/full_text.jsonl") 

    # Extract the final answer using a regex that looks for "[Final_Answer: NUMBER]"
    match = re.search(r"\[Final_Answer:\s*([0-2])\]", full_text)
    if match:
        final_answer = match.group(1)
    else:
        # Fallback: return the full text if the final answer is not found.
        final_answer = full_text.strip()

    print(f"Final Answer: {final_answer}")
    return final_answer