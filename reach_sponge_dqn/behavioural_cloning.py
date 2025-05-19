# Behavioural cloning algorithm to initialise RL policy and speed up training
# Source: https://stable-baselines.readthedocs.io/en/master/guide/pretrain.html

import numpy as np
import argparse
import json

from perception import Perception
from generate_poses import GeneratePoses

from pyrcareworld.envs.bathing_env import BathingEnv
import pyrcareworld.attributes.camera_attr as attr

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
    
    # Obstacles in the environment - from the perception system
    bedXL = -1.275
    bedXR =  1.275
    bedZT = 0.775
    bedZB = -0.775
    drawerXL = -0.315
    drawerXR = 0.638
    drawerZT = 2.775
    drawerZB = 1.726
    prev_distance = np.linalg.norm(np.array(sponge.data.get("position")) - np.array(robot.data.get("position")))
    # sponge_location - goal position
    sponge_position = np.array(sponge.data.get("position"), dtype=np.float32)
    env.step()
    
        
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
    poses = GeneratePoses().train_poses
    
    # For speed and efficiency, the initialisations will be done one after the other and then saved in a file.
    # Later, they'll be combined to generate the complete expert data
    robot.SetPosition(poses[15][0])
    robot.SetRotation(poses[15][1])
    
    print({
        "psition": poses[15][0],
        "rotation": poses[15][1]
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
    
    
    #  Add the very first the new observation to the trajectories array
    robot_position = np.array(robot.data.get("position"), dtype=np.float32)
    robot_rotation = np.array(robot.data.get("rotation"), dtype=np.float32)
        
    robotX = robot_position[0]
    robotZ = robot_position[2]
    spongeX = sponge_position[0]
    spongeZ = sponge_position[2]
    
    robotdx = robotX - spongeX
    robotdz = robotZ - spongeZ
    robotRy = robot_rotation[1]
    # Use minimum distance for safe obstacle avoidance
    robotd2bedx = min(np.abs(robotX-bedXL), np.abs(robotX-bedXR))
    robotd2bedz = min(np.abs(robotZ-bedZT), np.abs(robotZ-bedZB))
    robotd2drawerx = min(np.abs(robotX-drawerXL), np.abs(robotX-drawerXR))
    robotd2drawerz = min(np.abs(robotZ-drawerZT), np.abs(robotZ-drawerZB))
            
    observations.append([robotdx, robotdz, robotRy, robotd2bedx, robotd2bedz, robotd2drawerx, robotd2drawerz])
    
        
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
            env.step(225)
            update_obs = True   
         # Turn Right
        elif user_input == 2:  
            robot.TurnRight(90, 1.5)
            env.step(225)
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
                
            robotX = robot_position[0]
            robotZ = robot_position[2]
            spongeX = sponge_position[0]
            spongeZ = sponge_position[2]
            
            robotdx = robotX - spongeX
            robotdz = robotZ - spongeZ
            robotRy = robot_rotation[1]
            # Use minimum distance for safe obstacle avoidance
            robotd2bedx = min(np.abs(robotX-bedXL), np.abs(robotX-bedXR))
            robotd2bedz = min(np.abs(robotZ-bedZT), np.abs(robotZ-bedZB))
            robotd2drawerx = min(np.abs(robotX-drawerXL), np.abs(robotX-drawerXR))
            robotd2drawerz = min(np.abs(robotZ-drawerZT), np.abs(robotZ-drawerZB))
                    
            observations.append([robotdx, robotdz, robotRy, robotd2bedx, robotd2bedz, robotd2drawerx, robotd2drawerz])
        
            # Add the corresponding action
            actions.append(user_input)
                
            # check for collisions
            env.GetCurrentCollisionPairs()
            env.step()
            collision = env.data["collision_pairs"]
                
            # Calculate reward and check if episode is done
            reward, is_done = _get_reward(env, robot, collision, sponge)
                
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
    
    # Before dumping:
    serializable_data = {
        k: v.tolist() if isinstance(v, np.ndarray) else v
        for k, v in expert_data.items()
    }

    def make_json_safe(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.generic,)):  # e.g. np.float32, np.int64
            return obj.item()
        elif isinstance(obj, dict):
            return {k: make_json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_json_safe(v) for v in obj]
        else:
            return obj

    safe_data = make_json_safe(expert_data)
        
    # Append the new expert_data to a json lines file and save.
    with open("expert_traj.jsonl", "a") as f:
        json.dump(safe_data, f)
        f.write("\n")
        
    env.WaitLoadDone()
    
    
  
#  Turns not considered here for code simplicity
def _get_reward(env, robot, collision, sponge):
    robot_pos = robot.data.get("position")
    robot_rot = robot.data.get("rotation")
    prev_distance = np.linalg.norm(np.array(sponge.data.get("position")) - np.array(robot.data.get("position")))

    
    env.step()
        
    # Reward shaping.
    # Positive reward if the robot moves closer to the goal
    # Penalise timesteps to encourage efficiency
    prev_dist = prev_distance  
    curr_dist = np.linalg.norm(np.array(sponge.data.get("position")) - np.array(robot_pos))
    dist_reward = (prev_dist - curr_dist) * 10
    time_reward = -0.5 # small penalty to discourage non-progress
    reward = dist_reward + time_reward
    prev_distance = curr_dist
        
    x_low = -2.0
    x_high = 2.0
    z_low = -1.0
    z_high = 3.0

    truncated = False
    is_done = False
    is_success = False 
            
    # If the robot tries to go off-bounds, Penalise the robot and truncate the episode
    # My current action space doesnot have a Move_back action, so out of bounds problems may not be easy to recover from.
    if (robot_pos[0] > x_high or robot_pos[0] < x_low or robot_pos[2] > z_high or robot_pos[2] < z_low):
        reward = reward - 10
        truncated = True
            
    # Penalise the robot and end the episode if there is a collision
    # The robot might get stuck or fall. Punish this severely.
    if len(collision) > 0:
        reward = reward - 10
        is_done = True
            
    # Check if the robot is in the goal area
    full_goal, partial_goal = is_in_goal_area(robot_pos, robot_rot)
    if full_goal:
        print("------ robot is in goal area  ------")
        reward = reward + 30
        is_success = True
        is_done = True
            
    # # Truncate after 40 steps per episode (max_episode length is 12)
    # if n_eps_steps >= 40:
    #     truncated = True
    #     reward += -1 * np.linalg.norm(np.array(sponge_position) - np.array(robot_pos))
            
    # returns += reward
        
    return (reward, is_done)


#  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# Check if the robot is in the goal area for easy grasping
def is_in_goal_area(robot_pos, robot_rot):
    # Define the goal area bounds
    xmax, xmin = -0.050, -0.258
    # ymin, ymax = 0, 0  # ignore 
    zmax, zmin = 1.725, 1.400
        
    # Define valid rotation ranges
    xrot = {355, 356, 357, 358, 359}
    yrot = {260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280}
    zrot = {0, 358, 359, 360}
        
    # Extract robot position and rotation
    x = robot_pos[0]
    y = robot_pos[1]
    z = robot_pos[2]
    rx = int(robot_rot[0])
    ry = int(robot_rot[1])
    rz = int(robot_rot[2])
        
    # Check position constraints
    valid_pos = xmin <= x <= xmax and zmin <= z <= zmax
        
    # Check rotation constraints
    valid_rot = rx in xrot and ry in yrot and rz in zrot
        
    return (valid_pos and valid_rot, valid_pos and not valid_rot)
            
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run RCareWorld bathing environment simulation.')
    parser.add_argument('-g', '--graphics', action='store_true', help='Enable graphics')
    parser.add_argument('-d', '--dev', action='store_true', help='Run in developer mode')
    args = parser.parse_args()
    _main(use_graphics=args.graphics, dev=args.dev)