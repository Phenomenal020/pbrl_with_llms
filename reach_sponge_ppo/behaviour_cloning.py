# Behavioural cloning algorithm to initialise RL policy and speed up training
# Source: https://stable-baselines.readthedocs.io/en/master/guide/pretrain.html

import numpy as np
import argparse
import json

from perception import Perception
from generate_poses import GeneratePoses

from pyrcareworld.envs.bathing_env import BathingEnv
import pyrcareworld.attributes.camera_attr as attrz

def _main(use_graphics=False, dev=None):
        
    # Initialise the environment
    env = BathingEnv(graphics=use_graphics) if dev == False else BathingEnv(graphics=use_graphics, executable_file="@editor")
    
    # setup the camera and align the view to match the camera's view
    p = Perception(env)
    env.AlignCamera(p.scene_camera.id)

    # Get the robot, gripper and sponge (directly from the simulation environment or perception - doesn't matter)
    robot = env.get_robot()
    gripper = env.get_gripper()
    sponge = env.get_sponge()
    gripper_pos = gripper.data.get("position")
    
    # Advance the simulation to reflect changes
    env.step()
    
    # Setup trajectory arrays
    observations = []
    actions = []
    rewards = []
    episode_returns = []
    episode_starts = [True]
    dones = []
    info = []
    
    # Get the pretrain poses
    poses = GeneratePoses().pretrain_poses
    
    # For speed and efficiency, the initialisations will be done one after the other and then saved in a file.
    # Later, they'll be combined to generate the complete expert data
    robot.SetPosition(poses[14][0])
    robot.SetRotation(poses[14][1])
    
    print({
        "psition": poses[14][0],
        "rotation": poses[14][1]
    })
    
    # Open and raise the gripper to a safe height for easy obstacle avoidance and a lower action space 
    gripper.GripperOpen()
    robot.WaitDo()
    env.step()
        
    gripper_pos = [robot.data["position"][0], sponge.data.get("position")[1] + 0.3, robot.data["position"][2]]
    robot.IKTargetDoMove(
        position=gripper_pos,
        duration=1,
        speed_based=False,
    )
    robot.WaitDo()
    
    
    # Add the very first the new observation to the trajectories array
    robot_position = np.array(robot.data.get("position"), dtype=np.float32)
    robot_rotation = np.array(robot.data.get("rotation"), dtype=np.float32)
    # Concatenate all components into a single flat array.
    flattened_observation = np.concatenate([
        robot_position, 
        robot_rotation
    ])
    observations.append(flattened_observation.tolist())
        
    # Play game: Manually drive the robot to goal position
    done = False
    while not done:
         
        update_obs = False  
          
        # Take an action and update the environment
        try:
            user_input = int(input("Select robot action: 0 --> MoveForward; 1 --> TurnLeft; 2 --> TurnRight; "))
        except ValueError:
            print("Invalid input! Please enter a number.")
            
                
        # Move Forward
        if user_input == 0:  
            robot.MoveForward(0.2, 2)
            env.step(50)
            update_obs = True  
        # Turn Left
        elif user_input == 1:  
            robot.TurnLeft(90, 1.5)
            env.step(200)
            update_obs = True   
         # Turn Right
        elif user_input == 2:  
            robot.TurnRight(90, 1.5)
            env.step(200)
            update_obs = True 
        elif user_input == 3:
             # Lower gripper (Grasp sponge)
            lower_position = [sponge.data.get("position")[0], sponge.data.get("position")[1]+0.03, sponge.data.get("position")[2]]
            robot.IKTargetDoMove(
                position=lower_position,  # Move directly above the sponge to grasp it
                duration=1,
                speed_based=False,
            )
            robot.WaitDo()
            env.step(50)    
            gripper.GripperClose()
            env.step(50)
        else:
            print("Invalid input")
            update_obs = False
               
        # If a valid action is entered, add it to the actions list 
        if update_obs == True:
            # Add the new observation to the trajectories array
            robot_position = np.array(robot.data.get("position"), dtype=np.float32)
            robot_rotation = np.array(robot.data.get("rotation"), dtype=np.float32) 
            # Concatenate all components into a single flat array.
            flattened_observation = np.concatenate([
                robot_position, 
                robot_rotation
            ])
            observations.append(flattened_observation.tolist())
        
            # Add the corresponding action
            actions.append(user_input)
                
            # check for collisions
            env.GetCurrentCollisionPairs()
            env.step()
            collision = env.data["collision_pairs"]
                
            # Calculate reward and check if episode is done
            reward, is_done = get_reward(env, robot, collision)
                
            # Add the reward, episode returns, 
            rewards.append(reward)
            episode_returns = [sum(rewards)]
                
            # Continue until done is true - termination, collision or goal achievement
            done = is_done
            dones.append(done)
            info.append({})
            
            if done != True:
                episode_starts.append(False)
                
                
    # # Convert lists to numpy arrays when episode is done
    expert_data = {
        "obs": observations,
        "acts": actions,
        "rewards": rewards,
        "episode_returns": episode_returns,
        "episode_starts": episode_starts,
        "dones": dones,
        "info": info
    }
        
    print(f"expert data: {expert_data}")
        
    # Append the new expert_data to a json lines file and save.
    with open("expert_traj.jsonl", "a") as f:
        json.dump(expert_data, f)
        f.write("\n")
        
    env.WaitLoadDone()
    
  
def get_reward(env: BathingEnv, robot, collision):
  
    reward = -0.5 # To encourage task efficiency
    is_done = False
    x_low = -2.5
    x_high = 2.5
    z_low = -3.0
    z_high = 3.0  # So the robot doesn't wander too far from the target.
    robot_pos = robot.data.get("position")  
    env.step()
    
    # If the robot tries to go off-bounds, always turn left ( to speed up training )
    if (robot_pos[0] > x_high or robot_pos[0] < x_low or robot_pos[2] > z_high or robot_pos[2] < z_low):
        robot.TurnLeft(90, 1.5)
        env.step(240)

    # Penalise the robot and end the episode if there is a collision
    if len(collision) > 0:
        reward = -5
        is_done = True
        
    # Check if the robot is in the goal area
    if is_in_goal_area(robot):
        print("------ in goal area ------")
        reward = +20
        is_done = True
        
    print({
        "psition": robot.data.get("position"),
        "rotation": robot.data.get("rotation")
    })
        
    return (reward, is_done)


# Check if the robot is in the goal area for easy grasping
def is_in_goal_area(robot):
    # Define the goal area bounds
    xmax, xmin = -0.050, -0.258
    # ymin, ymax = 0, 0  # ignore 
    zmax, zmin = 1.725, 1.400
    
    # Define valid rotation ranges
    xrot = {355, 356, 357, 358, 359}
    yrot = {260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280}
    zrot = {0, 358, 359, 360}
    
    # Extract robot position and rotation
    x = robot.data.get("position")[0]
    y = robot.data.get("position")[1]
    z = robot.data.get("position")[2]
    rx = int(robot.data.get("rotation")[0])
    ry = int(robot.data.get("rotation")[1])
    rz = int(robot.data.get("rotation")[2])
    
    # Check position constraints
    valid_pos = xmin <= x <= xmax and zmin <= z <= zmax
    
    # Check rotation constraints
    valid_rot = rx in xrot and ry in yrot and rz in zrot
    
    return valid_pos and valid_rot


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run RCareWorld bathing environment simulation.')
    parser.add_argument('-g', '--graphics', action='store_true', help='Enable graphics')
    parser.add_argument('-d', '--dev', action='store_true', help='Run in developer mode')
    args = parser.parse_args()
    _main(use_graphics=args.graphics, dev=args.dev)