# # Behavioural cloning algorithm to speed up training for dip sponge in water tank without knocking it over

# # TBC...

# from pyrcareworld.envs.bathing_env import BathingEnv
# import numpy as np
# import cv2
# import argparse
# import random

# from Perception import Perception

# import pyrcareworld.attributes.camera_attr as attr


# def _main(use_graphics=False, dev=None):
        
#     # Initialise the environment
#     env = BathingEnv(graphics=use_graphics) if dev == False else BathingEnv(graphics=use_graphics, executable_file="@editor")
    
#     # setup the camera and align the view to match the camera's view
#     p = Perception(env)
#     env.AlignCamera(p.scene_camera.id)

#     # Get the robot, gripper and sponge (directly from the environment or perception - doesn't matter)
#     robot = env.get_robot()
#     gripper = env.get_gripper()
#     sponge = env.get_sponge()
#     gripper_pos = gripper.data.get("position")
    
#     # Note the sponge's initial position for grasping logic
#     sponge_initial_position = sponge.data.get("position")
    
#     # Advance the simulation to reflect changes
#     env.step()
    
#     # Define possible values for positions and rotations
#     x_values = np.linspace(-0.1, -0.2, num=11)  
#     y_pos = 0
#     z_values = np.linspace(1.62, 1.72, num=11) 

#     x_rot_values = [356, 357]
#     y_rot_values = [265, 266] 
#     z_rot_values = [0, 359, 360]

#     # Generate all possible combinations of positions and rotations
#     combinations = [
#         ([x, y_pos, z], [x_rot, y_rot, z_rot])
#         for x in x_values
#         for z in z_values
#         for x_rot in x_rot_values
#         for y_rot in y_rot_values
#         for z_rot in z_rot_values
#     ]

#     # Randomly sample 5 unique combinations for BC initialisation
#     # A human is the Teacher
#     random.seed(42)  # For reproducibility
#     random_combinations = random.sample(combinations, 5)

#     # Prepare data for use
#     positions = [pos for pos, _ in random_combinations]
#     rotations = [rot for _, rot in random_combinations]
    
#     # # Initialise trajectory data
#     observations = []
#     actions = []
#     rewards = []
#     episode_returns = []
#     episode_starts = []
    
#     # For each position and rotation combination:
#     for i in range(len(positions)):
#         # Intialise robot pose
#         robot.SetPosition(positions[i])
#         robot.SetRotation(rotations[i])
    
#         # Open the gripper - The gripper should be open at this point regarless
#         gripper.GripperOpen()
#         robot.WaitDo()
#         env.step(50)
        
#         # Initialise arrays for episode observations, actions, rewards, returns, and starts
#         episode_observations = []
#         episode_actions = []
#         episode_rewards = []
#         this_episode_returns = []
#         this_episode_starts = [True]
        
#         # Play game: Manually close the gripper to grasp the sponge, then dip in water tank and move towards mannequin 
#         done = False
#         repeat = False  # TODO: Logic to only add acceptable/efficient trajectries
        
        
#         # On start, set everything to zero
#         while not done:
#             # Add the new observation to the observations array
#             episode_observations.append({
#                 "gripper_position": np.array(gripper.data.get("position"), dtype=np.float32),
#                 "gripper_rotation": np.array(gripper.data.get("rotation"), dtype=np.float32), 
#                 "robot_position": np.array(robot.data.get("position"), dtype=np.float32),
#                 "robot_rotation": np.array(robot.data.get("rotation"), dtype=np.float32),  
#             })
#             update_obs = True
            
#             # Take an action and update the environment
#             try:
#                 user_input = int(input("Select robot action: 0 --> LowerGripper; 1 --> RaiseGripper, 2 --> MoveBack, 3 --> TurnRight: "))
#             except ValueError:
#                 print("Invalid input! Please enter a number.")
                
#             if user_input == 0:  
#                 # Lower gripper
#                 lower_position = [sponge.data.get("position")[0], sponge.data.get("position")[1]+0.03, sponge.data.get("position")[2]]
#                 robot.IKTargetDoMove(
#                     position=lower_position,  # Move directly above the sponge to grasp it
#                     duration=1,
#                     speed_based=False,
#                 )
#                 robot.WaitDo()
#                 env.step(50)
                
#             elif user_input == 1:  
#                 # Raise Gripper
#                 gripper_pos = [robot.data["position"][0], sponge.data.get("position")[1] + 0.5, robot.data["position"][2]]
#                 robot.IKTargetDoMove(
#                     position=gripper_pos,
#                     duration=0,
#                     speed_based=False,
#                 )
#                 robot.WaitDo()
#                 env.step(50)
                
#             elif user_input == 2:
#                 # Move back
#                 robot.MoveBack(0.3, 1)  # Predefined distance between sponge and center of water tank
#                 env.step(100)
                
#             elif user_input == 3:
#                 # Turn right
#                 robot.TurnRight(90, 1)
#                 env.step(300)
                
#             else:
#                 print("Invalid input")
#                 update_obs = False # on invalid input, do not add this observation
                
               
#             # If a valid action is entered, add it to the actions list 
#             if update_obs == True:
#                 episode_actions.append(user_input)
                
#                 # check for collisions
#                 env.GetCurrentCollisionPairs()
#                 env.step()
#                 collision = env.data["collision_pairs"]
                
#                 # Calculate reward and check if episode is done
#                 reward, is_done = get_reward(env, robot, collision, sponge_initial_position, sponge)
                
#                 episode_rewards.append(reward)
#                 this_episode_returns = sum(episode_rewards)
#                 this_episode_starts.append(False)
                
#                 done = is_done
                
#                 rewards.append(episode_rewards)
#                 observations.append(episode_observations)
#                 actions.append(episode_actions)
#                 episode_returns.append(this_episode_returns)
#                 episode_starts.append(this_episode_starts)
                
#         # # Convert lists to numpy arrays
#         expert_data = {
#             "obs": np.array(observations),
#             "actions": np.array(actions),
#             "rewards": np.array(rewards),
#             "episode_returns": np.array(episode_returns),
#             "episode_starts": np.array(episode_starts),
#         }
        
#         print(f"expert data: {expert_data}")

#     env.WaitLoadDone()
    
  
  
# def get_reward(env: BathingEnv, robot, collision, sponge_initial_position, sponge):
  
#     reward = -1 # To encourage task efficiency
#     is_done = False
#     x_low = -2.0
#     x_high = 2.0
#     z_low = -1.25
#     z_high = 2.25
#     robot_pos = robot.data.get("position")  
#     robot_rot = robot.data.get("rotation")  
#     env.step()
    
#     already_rewarded = False
    
#     print(f"robot's position ------------- {robot_pos}")
#     print(f"robot's orientation --------- {robot_rot} ")
    
#     # If the robot tries to go off-bounds, always turn right ( to speed up training ) - Highly unlikely
#     if (robot_pos[0] > x_high or robot_pos[0] < x_low or robot_pos[2] > z_high or robot_pos[2] < z_low):
#         robot.TurnRight(90, 1)
#         env.step(300)

#     # Penalise the robot and end the episode if there is a collision (with the water tank or drawer)
#     if len(collision) > 0:
#         reward = -10
#         is_done = True
        
#     if not already_rewarded:
#     # check if the sponge is being grasped
#     is_sponge_grasped, already_rewarded = _is_sponge_grasped(sponge_initial_position, sponge)
#     if is_sponge_grasped:
#         reward = 5
#         already_rewarded = True
        
        
#     # Check if the robot is in the sweet spot
#     if is_robot_in_sweet_spot(robot_pos, robot_rot):
#         print("robot is in the sweet spot")
#         reward = +10
#         is_done = True
        
#     return (reward, is_done)

# # TODO: replace with SpongeGrasper. Not working at the moment
# def _is_sponge_grasped(initial_sponge_pos, sponge):
#     return initial_sponge_pos != sponge.data.get("position")  # Sponge is assumed grasped if the current position is not equal to the initial position

# # TODO: replace with a better method
# # Sponge is considered soaked if at anytime, sponge position coincides with the water tank's position
# # def is_sponge_soaked(sponge):
#     # return sponge.data.get("position") 
 
# # REward the robot if it is close to the manikin and sponge_grasped is true and sponge_wet is true   
# def is_robot_near_manikin(robot, sponge_grasped, sponge_soaked):
#     pass
    


# # Check if the robot is close to the mannequin
# def is_robot_in_sweet_spot(robot_pos, robot_rot):
#     position_check = (-0.20 <= robot_pos[0] <= -0.10) and (1.45 <= robot_pos[2] <= 1.65)
#     rot_x, rot_y, rot_z = map(int, robot_rot)  # convert rotation values are whole numbers

#     rotation_check = (rot_x in {356, 357, 265, 266}) and \
#                      (rot_y in {265, 266}) and \
#                      (rot_z in {359, 360, 0})
    
#     return position_check and rotation_check


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Run RCareWorld bathing environment simulation.')
#     parser.add_argument('-g', '--graphics', action='store_true', help='Enable graphics')
#     parser.add_argument('-d', '--dev', action='store_true', help='Run in developer mode')
#     args = parser.parse_args()
#     _main(use_graphics=args.graphics, dev=args.dev)
    

#     # TODO: Save after rigorous cloning (10+ episodes)
#     # TODO: Save the expert data to be used for pretraining later