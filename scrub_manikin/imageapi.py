import datetime
from montage_maker import ImageMontage
import cv2
import os

obsdir = f"enhanced_rl/observations"
os.makedirs("enhanced_rl/db/full_text/", exist_ok=True)
os.makedirs("enhanced_rl/montage/", exist_ok=True)

#  Sources: 
# https://github.com/TU2021/RL-SaLLM-F/blob/main/prompt.py
# https://gymnasium.farama.org/environments/toy_text/frozen_lake/

import json
import re
import numpy as np

import jsonlines
import json
import os
from datetime import datetime

import base64
from openai import OpenAI

client = OpenAI()

def traj_pair_process(obs1, obs2, acts1, acts2, truncated1, truncated2, goals1, goals2, imageobses1, imageobses2):
    # Check if the lengths of the trajectories are the same.
    if len(obs1) != len(obs2) or len(acts1) != len(acts2) or len(truncated1) != len(truncated2) or len(goals1) != len(goals2) or len(imageobses1) != len(imageobses2):
        raise ValueError("Trajectories must have the same length.")

    # Convert each component to a Python list if possible.
    traj1_data = {
        "observations": obs1.tolist() if hasattr(obs1, "tolist") else obs1,
        "actions": acts1.tolist() if hasattr(acts1, "tolist") else acts1,
        # "terminated": dones1.tolist() if hasattr(dones1, "tolist") else dones1,
        "truncated": truncated1.tolist() if hasattr(truncated1, "tolist") else truncated1,
        "goal": goals1.tolist() if hasattr(goals1, "tolist") else goals1,
        "imageobses1": imageobses1.tolist() if hasattr(imageobses1, "tolist") else imageobses1
    }
    traj2_data = {
        "observations": obs2.tolist() if hasattr(obs2, "tolist") else obs2,
        "actions": acts2.tolist() if hasattr(acts2, "tolist") else acts2,
        # "terminated": dones2.tolist() if hasattr(dones2, "tolist") else dones2,
        "truncated": truncated2.tolist() if hasattr(truncated2, "tolist") else truncated2,
        "goal": goals2.tolist() if hasattr(goals2, "tolist") else goals2,
        "imageobses2": imageobses2.tolist() if hasattr(imageobses2, "tolist") else imageobses2
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
        expert_actions_1 = get_nearest_expert_demo(traj1["observations"][0], traj1["goal"][0])
        expert_actions_2 = get_nearest_expert_demo(traj2["observations"][0], traj2["goal"][0])
        print(f"expert_actions_1: {expert_actions_1}")
        print(f"expert_actions_2: {expert_actions_2}")
        expert_actions_1_str = json.dumps(expert_actions_1, indent=2, default=numpy_encoder)
        expert_actions_2_str = json.dumps(expert_actions_2, indent=2, default=numpy_encoder)
    # Use the custom numpy_encoder in json.dumps.
    traj1_str = json.dumps(traj1, indent=2, default=numpy_encoder)
    traj2_str = json.dumps(traj2, indent=2, default=numpy_encoder)
    # image1, image2, image3, image4 = get_image_paths(traj1, traj2)
    if use_rag:
        return traj1_str, traj2_str, expert_actions_1_str, expert_actions_2_str
    return traj1_str, traj2_str


def get_image_paths(traj1, traj2):
    image1_start_path = traj1["imageobses1"][0]
    image1_end_path = traj1["imageobses1"][-1]
    image2_start_path = traj2["imageobses2"][0]
    image2_end_path = traj2["imageobses2"][-1]
    print(f"image1_start_path: {image1_start_path}")
    print(f"image1_end_path: {image1_end_path}")
    print(f"image2_start_path: {image2_start_path}")
    print(f"image2_end_path: {image2_end_path}")
    return image1_start_path, image1_end_path, image2_start_path, image2_end_path



def get_nearest_expert_demo(starting_obs, goal):
    # Load expert demonstrations
    expert_demos = []
    try:
        with open("expert_traj_action.jsonl", "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    demo = json.loads(line)
                    # require obs, acts, and goal keys
                    if all(k in demo for k in ("obs", "acts", "goal")):
                        expert_demos.append(demo)
                    else:
                        print("Skipping demo missing 'obs', 'acts', or 'goal'")
                except json.JSONDecodeError as e:
                    print(f"JSON decode error: {e} in line: {line}")
    except FileNotFoundError:
        print("Error: File expert_traj_action.jsonl not found.")
        return None
    except Exception as e:
        print("Error loading expert demos:", e)
        return None

    if not expert_demos:
        print("No expert demonstrations loaded.")
        return None

    # Ensure numpy arrays
    starting_obs = np.array(starting_obs)
    goal = np.array(goal)

    # Track best match across all demos
    min_dist = float('inf')
    best_actions_segment = None

    for demo in expert_demos:
        # Compare goals first (string equality)
        demo_goal = demo["goal"][0]
        if goal != demo_goal:
            # skip demos whose goal doesn't match exactly
            continue

        obs_list = demo["obs"]
        acts_list = demo["acts"]

        # Search for the closest observation in this demo
        for idx, obs in enumerate(obs_list):
            demo_obs = np.array(obs)
            dist = np.linalg.norm(starting_obs - demo_obs)
            if dist < min_dist:
                min_dist = dist
                best_actions_segment = acts_list[idx:]

    # Return up to 5 actions if found
    if best_actions_segment is not None:
        return best_actions_segment[:5]
    else:
        # No matching demo found
        return []


gpt_query_reach_sponge_rag = """Suppose you are an expert robot trajectory evaluator. You are provided with two trajectory segments from any two trajectories. Now you need to evaluate the quality of the robot's motion trajectory. You are provided with a montage of 4 images. The top frames are the starting and final sponge positions of trajectory  segment 1 while the bottom 2 frames are the starting and final sponge positions of trajectory segment 2. The environment is a care bathing environment where a robot is expected to scrub a manikin with a sponge as shown in the images. You should analyse the image carefully to determine which of the pair is more desirable. Consider priorities as (FIRST: did the sponge make contact with the manikin in the GOAL part? For example, if the goal is the arm, then evaluate whether the robot made contact with the arm. Note if the goal is the arm but the robot is in contact with the torso, then penalise). (EQUALLY IMPORTANT: is the sponge in a bad position? Is it beneath the arm? Way beyond the goal? If so, heavily penalise) (NEXT: consider how much the sponge has moved closer to the goal between frames of the trajectory. Reward good progress). (FINALLY: truncation - did the episode get truncated? This is not harmful if the robot is making progress towards the goal) in that order.
Action Space:
- Discrete(3) with the following actions: 
0 -> Move sponge 20 mm towards the manikin
1 -> Move sponge downwards by 50 mm
2 -> TMove sponge upwards by 50 mm

Target/Goal:
Indicated by the goal array in the input. This is one of the following: 'arm', 'torso', or 'leg'

Observation Space:
Use the image for observation. 

Truncated:
- The episode is truncated if the sponge takes too many steps to reach the mannequin (n >= 15) (discourage but not detrimental).

You are provided with two trajectory segments from a trajectory containing: 
(1) "actions" represents the sequence of actions taken by the agent.
(2) "goal" represents where the sponge is supposed to be. 
(3) "truncated" indicates whether the episode was truncated.

Please answer the following questions step by step: 
1. Describe what you see in the image.
2. Using what you just described,is there any difference between Trajectory 1 and Trajectory 2 in terms of achieving the goal? Which trajectory do you think better achieves the goal?  
2. Reply with a single number 1 if you think the goal is BETTER ACHIEVED in Trajectory 1, or 2 if it is BETTER ACHIEVED in Trajectory 2 and 0 if the text is unsure or there is no significant difference.
Format your response this way: [Final_Answer: 0 or 1 or 2]
"""


def gpt_infer_rag(traj_str, traj_str2, expert_actions_1_str, expert_actions_2_str,  image1, image2, image3, image4, prompt=gpt_query_reach_sponge_rag, index=None):
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
            
    mm = ImageMontage(grid_size_px=768, rows=2, cols=2)
    print(image1, image2, image3, image4)
    montage, image_path = mm.create_montage([f"{obsdir}/{image1}", f"{obsdir}/{image2}", f"{obsdir}/{image3}", f"{obsdir}/{image4}"], output_path=f"enhanced_rl/montage/{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png")
    base64_image = encode_image(image_path)
    if index == 1:
       response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {
                "role": "user",
                "content": [
                    { 
                        "type": "input_text", 
                        "text": f"{prompt}\n\nTrajectory 1:\n{traj_str}\n\nTrajectory 2:\n{traj_str2}\n\n Expert Actions 1:\n{expert_actions_1_str}\n\nExpert Actions 2:\n{expert_actions_2_str}\n\n"
                    },
                    {  
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "high"
                    },
                ],
            }
        ],
    )
    else:
        response = client.responses.create(
        model="o4-mini",
        input=[
            {
                "role": "user",
                "content": [
                    { 
                        "type": "input_text", 
                        "text": f"{prompt}\n\nTrajectory 1:\n{traj_str}\n\nTrajectory 2:\n{traj_str2}\n\n Expert Actions 1:\n{expert_actions_1_str}\n\nExpert Actions 2:\n{expert_actions_2_str}\n\n"
                    },
                    {  
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "high"
                    },
                ],
            }
        ],
        reasoning={
            "effort": "medium",
            "summary": "auto"
        },
        tools=[],
        store=False
    )
       
    print(f"using model: {response.model} with rag")

    # Get the full output text from the response
    full_text = response.output_text
    input_token_count = response.usage.input_tokens
    output_token_count = response.usage.output_tokens

    print(f"Full Text: {full_text}")
    print(f"Input Token Count: {input_token_count}")
    print(f"Output Token Count: {output_token_count}")

    with jsonlines.open("ablation/db/full_text/full_text.jsonl", mode='a') as writer:
        writer.write({"full_text": full_text, "input_token_count": input_token_count, "output_token_count": output_token_count})
        print("Saved segment to enhanced_rl/db/full_text/full_text.jsonl") 

    # Extract the final answer using a regex that looks for "[Final_Answer: NUMBER]"
    match = re.search(r"\[Final_Answer:\s*([0-2])\]", full_text)
    if match:
        final_answer = match.group(1)
    else:
        # Fallback: return the full text if the final answer is not found.
        final_answer = full_text.strip()

    print(f"Final Answer: {final_answer}")
    return final_answer

# =======================================================================================

gpt_query_reach_sponge_no_rag = """Suppose you are an expert robot trajectory evaluator. You are provided with two trajectory segments from any two trajectories. Now you need to evaluate the quality of the robot's motion trajectory. You are provided with a montage of 4 images. The top frames are the starting and final sponge positions of trajectory  segment 1 while the bottom 2 frames are the starting and final sponge positions of trajectory segment 2. The environment is a care bathing environment where a robot is expected to scrub a manikin with a sponge as shown in the images. You should analyse the image carefully to determine which of the pair is more desirable. 

Action Space:
- Discrete(3) with the following actions: 
0 -> Move sponge 20 mm towards the manikin
1 -> Move sponge downwards by 50 mm
2 -> Move sponge upwards by 50 mm

Target/Goal:
Indicated by the goal array in the input. This is one of the following: 'arm', 'torso', or 'leg'

Observation Space:
Use the image for observation. 

Truncated:
- The episode is truncated if the sponge takes too many steps to reach the mannequin (n >= 15) (discourage but not detrimental).

You are provided with two trajectory segments from a trajectory containing: 
(1) "actions" represents the sequence of actions taken by the agent.
(2) "goal" represents where the sponge is supposed to be. 
(3) "truncated" indicates whether the episode was truncated.

Please answer the following questions step by step: 
1. Describe what you see in the image.
2. Using what you just described,is there any difference between Trajectory 1 and Trajectory 2 in terms of achieving the goal? Which trajectory do you think better achieves the goal?  
2. Reply with a single number 1 if you think the goal is BETTER ACHIEVED in Trajectory 1, or 2 if it is BETTER ACHIEVED in Trajectory 2 and 0 if the text is unsure or there is no significant difference.
Format your response tas either: [Final_Answer: 0 or 1 or 2]
"""



def gpt_infer_no_rag(traj_str, traj_str2, image1, image2, image3, image4, prompt=gpt_query_reach_sponge_no_rag, index=None):
    print(f"{image1}, {image2}, {image3}, {image4}")
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
            
    mm = ImageMontage(grid_size_px=768, rows=2, cols=2)
    montage, image_path = mm.create_montage([f"{obsdir}/{image1}", f"{obsdir}/{image2}", f"{obsdir}/{image3}", f"{obsdir}/{image4}"], output_path=f"enhanced_rl/montage/{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png")
    base64_image = encode_image(image_path)
    if index == 1:
        response = client.responses.create(
            model="gpt-4o-mini",
            input=[
                {
                    "role": "user",
                    "content": [
                    { 
                        "type": "input_text", 
                        "text": f"{prompt}\n\nTrajectory 1:\n{traj_str}\n\nTrajectory 2:\n{traj_str2}\n\n"
                    },
                    {  
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "high"
                    },
                    ],
                },
            ],
            text={
                "format": {
                    "type": "text"
                }
            },
            reasoning={},
            tools=[],
            store=False,
            temperature=0.1,
            # max_output_tokens=2048,
       )
    else:
        response = client.responses.create(
        model="o4-mini",
        input=[
            {
                "role": "user",
                "content": [
                    { 
                        "type": "input_text", 
                        "text": f"{prompt}\n\nTrajectory 1:\n{traj_str}\n\nTrajectory 2:\n{traj_str2}\n\n "
                    },
                    {  
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "high"
                    },
                ],
            }
        ],
        reasoning={
            "effort": "medium",
            "summary": "auto"
        },
        tools=[],
        store=False
    )
        

    print(f"using model: {response.model} without rag")

    # Get the full output text from the response
    full_text = response.output_text
    input_token_count = response.usage.input_tokens
    output_token_count = response.usage.output_tokens

    with jsonlines.open("ablation/db/full_text/full_text.jsonl", mode='a') as writer:
        writer.write({"full_text": full_text, "input_token_count": input_token_count, "output_token_count": output_token_count})
        print("Saved segment to ablation/db/full_text/full_text.jsonl") 

    # Extract the final answer using a regex that looks for "[Final_Answer: NUMBER]"
    match = re.search(r"\[Final_Answer:\s*([0-2])\]", full_text)
    if match:
        final_answer = match.group(1)
    else:
        # else, return the full text if the final answer is not found.
        final_answer = full_text.strip()

    print(f"Final Answer: {final_answer}")
    return final_answer