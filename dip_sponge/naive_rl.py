# # Tutorial: https://github.com/empriselab/RCareWorld/blob/phy-robo-care/pyrcareworld/pyrcareworld/demo/examples/example_rl.py  

# # TODO: Yet to figure out sponge soaking logic due to issues with the api. Working on it.

# import numpy as np
# import os

# from project_code.perception import Perception

# try:
#     import gymnasium as gym
# except ImportError:
#     print("This feature requires gymnasium, please install with `pip install gymnasium`")
#     raise
# from gymnasium import spaces

# from pyrcareworld.envs.bathing_env import BathingEnv
# import pyrcareworld.attributes as attr
# from pyrcareworld.envs.base_env import RCareWorld

# # Import A2C and other utilities from stable_baselines3
# from stable_baselines3 import A2C
# from stable_baselines3.common.vec_env import DummyVecEnv
# from stable_baselines3.common.monitor import Monitor


# class DipSponge(gym.Env):	
    
#     # Initialisation function
#     def __init__(self, use_graphics=False, dev=None):
        
#         super(DipSponge, self).__init__()

#         # Initialize the bathing environment in headless mode 
#         self.env = BathingEnv(graphics=use_graphics) if not dev else BathingEnv(graphics=use_graphics, executable_file="@editor")

#         #TODO: Remove this part later since I do not need it in the actual work.
#         # Grab a perception object to identify the location of objects in the environment
#         p = Perception(self.env)

#         # Get the robot instance in the current environment
#         self.robot = self.env.get_robot()
         
#         #  Get the gripper too
#         self.gripper = self.env.get_gripper()

#         #  Get the sponge
#         self.sponge = self.env.get_sponge()
#         self.env.step()
        
#         # Use the initial robot position for placeholder_is_sponge_being_held() 
#         self.initial_sponge_position = self.sponge.data.get("position")
#         self.sponge_grasp_reward_checker = False # Sponge grasp has not been rewarded yet.
#         self.sponge_has_been_grasped = False
        
#         self.sponge_soaked = False # Placeholder method to check if sponge has been dipped
#         self.sponge_soaked_reward_checker = False # sponge soak has not been rewarded yet
        
#         # Initialise an n_steps variable to keep track of number of steps taken per episode. 
#         self.n_steps = 0
         
#         #  define the action space - Discrete as recommended by the rcareworld authors
#         self.action_space = spaces.Discrete(4)        

#         # define the observation space.
#         self.observation_space = spaces.Dict({
#             "gripper_position": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
#             "gripper_rotation": spaces.Box(low=-1.0, high=360.0, shape=(3,), dtype=np.float32),
#             "robot_position": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
#             "robot_rotation": spaces.Box(low=-1.0, high=360.0,  shape=(3,), dtype=np.float32),
#         })
         
#         #  Randomise Robot's initial position and raise the gripper on initialisation
        
#         # Define possible values for positions and rotations - Graspable positions from the sponge. If reach_sponge is successful, this should always be the case
         

#         # Define possible x values and choose one randomly
#         x_values = np.linspace(-0.1, -0.2, num=11) 
#         x = np.random.choice(x_values)
#         # y is fixed
#         y_pos = 0
#         # Define possible z values and choose one randomly
#         z_values = np.linspace(1.62, 1.72, num=11) 
#         z = np.random.choice(z_values)
        
#         # DO same for the rotations
#         x_rot_values = np.array([356, 357])
#         y_rot_values = [265, 266] 
#         z_rot_values = [0, 359, 360]
#         x_rot = np.random.choice(x_rot_values)
#         y_rot = np.random.choice(y_rot_values)
#         z_rot = np.random.choice(z_rot_values)
        
#         # Set the robot's position and rotation
#         self.robot.SetPosition([x, y_pos, z])
#         self.robot.SetRotation([x_rot, y_rot, z_rot])  # on intitialisation
    
         
#         # Open the gripper - The gripper should be open at this point regarless
#         self.gripper.SetRotation(self.robot.data.get("rotation"))
#         self.gripper.GripperOpen()
#         self.robot.IKTargetDoComplete()
#         self.robot.WaitDo()
#         self.env.step(50)
        
      
#     #   +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      
      
      
#     # Whenever the environment is reset (after an episode), initialise the robot, gripper, and return an observation 
#     def reset(self, seed=None, options=None):
        
#         # Handle the seed if provided - for reproducibility
#         if seed is not None:
#             self.np_random, seed = gym.utils.seeding.np_random(seed)
            
#         # Define possible x values and choose one randomly
#         x_values = np.linspace(-0.1, -0.2, num=11) 
#         x = np.random.choice(x_values)
#         # y is fixed
#         y_pos = 0
#         # Define possible z values and choose one randomly
#         z_values = np.linspace(1.62, 1.72, num=11) 
#         z = np.random.choice(z_values)
        
#         # DO same for the rotations
#         x_rot_values = np.array([356, 357])
#         y_rot_values = [265, 266] 
#         z_rot_values = [0, 359, 360]
#         x_rot = np.random.choice(x_rot_values)
#         y_rot = np.random.choice(y_rot_values)
#         z_rot = np.random.choice(z_rot_values)
        
#         # Set the robot's position and rotation
#         self.robot.SetPosition([x, y_pos, z])
#         self.robot.SetRotation([x_rot, y_rot, z_rot])  # on intitialisation
    
         
#         # Open the gripper - The gripper should be open at this point regarless
#         self.gripper.SetRotation(self.robot.data.get("rotation"))
#         self.gripper.GripperOpen()
#         self.robot.IKTargetDoComplete()
#         self.robot.WaitDo()
#         self.env.step(50)
        
#         # reset the number of steps taking in the episode
#         self.n_steps = 0
        
#         observation = self._get_observation()
#         print("++++++++++++++")
#         print("resetting")
#         return observation, None 
    
    
#     # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    
    
#     # Each step in an episode
#     def step(self, action):
        
#         # Apply the selected action
#         self._perform_action(action)

#         # Get the updated observation
#         observation = self._get_observation()

#         # Compute the reward based on the new state
#         reward, terminated, truncated = self._compute_reward()
        
#         info = {}

#         # Return the step tuple: observation, reward, done, and additional info
#         return observation, reward, terminated, truncated, info
    
    
#     # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    
    
    
#     def render(self):
#         pass
    
    
#     # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    
    
#     def close(self):
#         self.env.close()
        
        
#     # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        
    
    
#     # Helper methods 
#     # ------------------------------------------------------------------------------
    
#     def _perform_action(self, action):
#         if action == 0:  
#             # Lower gripper
#             lower_position = [self.initial_sponge_position[0], self.initial_sponge_position[1]+0.03, self.initial_sponge_position[2]]
#             self.robot.IKTargetDoMove(
#                 position=lower_position,  # Move directly above the sponge to grasp it
#                 duration=1,
#                 speed_based=False,
#             )
#             self.robot.WaitDo()
#             self.env.step(50)
                        
#         elif action == 1:  
#             # Raise Gripper
#             gripper_pos = [self.robot.data["position"][0], self.initial_sponge_position[1] + 0.5, self.robot.data["position"][2]]
#             self.robot.IKTargetDoMove(
#                 position=gripper_pos,
#                 duration=1,
#                 speed_based=False,
#             )
#             self.robot.WaitDo()
#             self.env.step(50)
                        
#         elif action == 2:
#             # Move back
#             self.robot.MoveBack(0.15, 1) 
#             self.env.step(50)
                        
#         elif action == 3:
#             # Turn right
#             self.robot.TurnRight(90, 1)
#             self.env.step(300)
                        
#         else:
#             pass
        
    
#     # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        
        
        
#     def _get_observation(self):
#         return {
#             "gripper_position": np.array(self.gripper.data.get("position"), dtype=np.float32),
#             "gripper_rotation": np.array(self.gripper.data.get("rotation"), dtype=np.float32),
#             "robot_position": np.array(self.robot.data.get("position"), dtype=np.float32),
#             "robot_rotation": np.array(self.robot.data.get("rotation"), dtype=np.float32),
#         }
        
    
#     # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        
    
    
#     def _compute_reward(self):
#         # Check for collisions
#         self.env.GetCurrentCollisionPairs()
#         self.env.step()
#         collision = self.env.data["collision_pairs"]
        
#         # Get reward and whether the episode is over
#         reward, is_done, truncated = self._get_reward(collision)
#         return reward, is_done, truncated
    
    
#     # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    
    
#     def _get_reward(self, collision):
    
#         reward = -1 # To encourage task efficiency
        
#         is_done = False
        
#         x_low = -2.25
#         x_high = 2.25
#         z_low = -1.75
#         z_high = 2.75
        
#         robot_pos = self.robot.data.get("position")
#         robot_rot = self.robot.data.get("rotation")
        
#         self.env.step()
        
#         truncated = False
        
#         # Increase the number of steps per episode
#         self.n_steps = self.n_steps + 1
        
#         # If the robot tries to go off-bounds, always turn right ( to speed up training )
#         if (robot_pos[0] > x_high or robot_pos[0] < x_low or robot_pos[2] > z_high or robot_pos[2] < z_low):
#             self.robot.TurnRight(90, 1)
#             env.step(300)
            
#         # Penalise the robot and end the episode if there is a collision
#         if len(collision) > 0:
#             reward = -5
#             is_done = True
            
#     #    Check if the sponge is grasped and the robot has been reward once
#         reward = self._is_sponge_grasped()
#         # reward = self._is_sponge_soaked() TODO: yet to find a soaking logic. Still scanning the api. Otherwise, would have to use the Perception class.
            
#         # Truncate after 100 steps per episode
#         if self.n_steps >= 100:
#             truncated = True
#             is_done = True
            
#         print({
#             "n_steps": self.n_steps,
#             "reward": reward
#         })   
#         return (reward, is_done, truncated)
    
    
#     # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



#     # TODO: How to check for soaked sponge. Issues with the api methods
#     # Check if sponge has been soaked and reward only once.
#     def _is_sponge_soaked(self):
#         if not self.sponge_has_been_grasped: # sponge can not be soaked without getting grasped first
#             reward = -1
#         elif self.sponge_has_been_grasped and self.sponge_soaked and not self.sponge_soaked_reward_checker: # sponge has been grasped and is now soaked and has not been rewarded
#             reward = +5
#             self.sponge_soaked_reward_checker = True
#         else:
#             reward = -1 # IN any other case, reward -1.
#         return reward
            
            
#     # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
#     # check if sponge has been grasped and reward only once
#     def _is_sponge_grasped(self):
#         # if these are unequal, then the sponge has been grasped
#         sponge_has_been_grasped = self.initial_sponge_position != self.sponge.data.get("position")
#         # check if this action has already been rewarded
#         # If this is true, then this action has been rewarded already. Return no further rewards
#         if sponge_has_been_grasped and self.sponge_grasp_reward_checker: 
#             reward = -1
#         elif sponge_has_been_grasped and not self.sponge_grasp_reward_checker:
#             reward = 5
#             self.sponge_grasp_reward_checker = True
#             self.sponge_has_been_grasped = True
#         else:
#             reward = -1 # Sponge is not being grasped
#         return reward
              
# #    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


#     # TODO: Fix sponge soaked and test method.
#     def _is_robot_at_goal(self):
#         reward = -1
#         # If the robot is approaching this position with the sponge grasped and wet, 
#         if self.sponge_has_been_grasped and self._is_sponge_soaked:
#             position_check = (0.21 <= self.robot.data.get("position")[0] <= 0.31) and (0.9 <= self.robot.data.get("position")[0]  <= 1.1)
#             rot_x, rot_y, rot_z = map(int, self.robot.data.get("rotation"))  # convert rotation values are whole numbers
#             rotation_check = (rot_x in {356, 357}) and (rot_y in {87, 88}) and (rot_z in {359, 360, 0})
#             if rotation_check and position_check:
#                 reward = +10
#         return reward
    
#     #    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        
                   
# if __name__ == "__main__":
    
#     env = DipSponge()
#     env = Monitor(env)
#     env = DummyVecEnv([lambda: env])
    
#     # dirs for my models
#     models_dir = "models/a2c"
#     logdir = "logs"
#     if not os.path.exists(models_dir):
#         os.makedirs(models_dir)
#     if not os.path.exists(logdir):
#         os.makedirs(logdir)

#     # Create an A2C agent and use GPU
#     # TODO: Setup Cuda
#     model = A2C("MultiInputPolicy", env, verbose=1, device='cuda', tensorboard_log=logdir) 

#     # Train the agent - 50,000 timesteps and save at 1,000 timesteps interval
#     TIMESTEPS = 50000
#     SAVE_INTERVAL = 1000

#     # Train the model incrementally and save it at regular intervals
#     for i in range(0, TIMESTEPS, SAVE_INTERVAL):
#         model.learn(total_timesteps=SAVE_INTERVAL, reset_num_timesteps=False, tb_log_name="a2c")
#         model.save(f"{models_dir}/dip_sponge_a2c{i + SAVE_INTERVAL}")
#         print(f"Model saved at timestep {i + SAVE_INTERVAL}")

#     # Optionally, load and evaluate any agent
#     # model = A2C.load(f"{models_dir}/reach-sponge_a2c_{5000}")
#     # obs = env.reset()
#     # episodes = 100
#     # for _ in range(episodes):
#     #     action, _states = model.predict(obs)
#     #     obs, rewards, done, info = env.step(action)
#     #     print(rewards)
#     #     if done:
#     #         obs = env.reset()

#     env.close()