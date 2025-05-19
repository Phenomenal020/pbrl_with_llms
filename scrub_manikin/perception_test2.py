# Perception class:
# To get the locations of objects in the scene.
# This file was mostly used during the initial stages to know the locations of objects. Subsequently, the code works fine without it.

from os import preadv
from pyrcareworld.envs.base_env import RCareWorld
from pyrcareworld.envs.bathing_env import BathingEnv
import pyrcareworld.attributes.camera_attr as attr
import numpy as np
import cv2

import mediapipe as mp 
import json
import time
import jsonlines

from detect_pose import DetectPose

from generate_trajectories import GenTrajectory

# import pyrcareworld.attributes as _attr
from pyrcareworld.attributes.omplmanager_attr import OmplManagerAttr
from pyrcareworld.attributes import PersonRandomizerAttr


# PDController 
# from pid import PDController

try:
    import pyrcareworld.attributes.omplmanager_attr as rfu_ompl
except ImportError:
    raise Exception("This feature requires ompl, see: https://github.com/ompl/ompl")


    
class ReachMannequin:
    def __init__(self, env:BathingEnv, robot, gripper, sponge, trajectories):
        self.env = env  
        self.robot = robot
        self.gripper = gripper
        self.sponge = sponge
        self.trajectories = trajectories
        self.initial_sponge_pos = self.sponge.data.get("position")
        self.env.step()
        
        person_randomizer = PersonRandomizerAttr(self.env, 573920)
        person_randomizer.SetSeed(42)
        self.env.step()
        
        self.person_midpoint = [0.055898, 0.84, 0.003035]  # initialise with a confirmed midpoint
                
        # Setup a camera to capture the entire scene
        self.env.scene_camera = self.env.InstanceObject(name="Camera", id=888888, attr_type=attr.CameraAttr) 
        # Set position and orientation for the scene camera
        self.env.scene_camera.SetTransform(position=[0, 4.2, 0.6], rotation=[90, 0, 0.0]) 
        self.env.scene_camera.Get3DBBox()
        self.env.step()
        for i in self.env.scene_camera.data["3d_bounding_box"]:
            if i == 573920:
                self.person_midpoint = self.env.scene_camera.data["3d_bounding_box"][i][0]
                print(f"Person midpoint: {self.person_midpoint}")
        self.env.step()
        
        
        # self.robot.SetPosition([0.19908380508422852, 0.009266123175621033, 1.05])
        # self.robot.SetRotation([357.8343505859375, 0, 359.7845764160156])
        # gripper_pos = [self.robot.data["position"][0], self.initial_sponge_pos[1] + 0.2, self.robot.data["position"][2]]
        # self.robot.IKTargetDoMove(
        #     position=gripper_pos,
        #     duration=1,
        #     speed_based=False,
        # )
        # self.robot.WaitDo()
        # self.env.step(50)
        # self.robot.TurnRight(90, 1.5)
        # self.env.step(250)
        
    def grasp_sponge(self):
         # Raise the gripper to a safe height for easy obstacle avoidance
        gripper_pos = [self.robot.data["position"][0], self.initial_sponge_pos[1] + 0.2, self.robot.data["position"][2]]
        self.robot.IKTargetDoMove(
            position=gripper_pos,
            duration=1,
            speed_based=False,
        )
        self.robot.WaitDo()
        self.env.step(50)
        self.gripper.GripperOpen()
        self.env.step(50)
        
        # Move towards sponge
        self.robot.MoveForward(0.25, 2)
        self.env.step(100)
        
        # Turn left
        self.robot.TurnLeft(90, 1.5)
        self.env.step(200)
        
        # Move forward 
        self.robot.MoveForward(0.2, 2)
        self.env.step(50)
        
        # Lower gripper (Grasp sponge)
        lower_position = [self.sponge.data.get("position")[0], self.initial_sponge_pos[1]+0.03, self.sponge.data.get("position")[2]]
        self.robot.IKTargetDoMove(
            position=lower_position,  
            duration=1,
            speed_based=False,
        )
        self.robot.WaitDo()
        self.env.step(50)
        
        # Grasp sponge
        self.gripper.GripperClose()
        self.env.step(50)
        
        # Raise Gripper
        self.robot.IKTargetDoMove(
            position=gripper_pos,
            duration=1,
            speed_based=False,
        )
        self.robot.WaitDo()
        self.env.step(50)
        
    
    def dip_sponge(self):
        # Move backwards
        self.robot.MoveBack(0.40, 2)
        self.env.step(50)
            
        # Lower Gripper to dip sponge
        lower_position = [self.sponge.data.get("position")[0], self.initial_sponge_pos[1]+0.03, self.sponge.data.get("position")[2]]
        self.robot.IKTargetDoMove(
            position=lower_position,  
            duration=1,
            speed_based=False,
        )
        self.robot.WaitDo()
        self.env.step(50)
        
        # Raise Gripper
        reference_gripper_pos = [self.robot.data["position"][0], self.initial_sponge_pos[1] + 0.5, self.robot.data["position"][2]]
        self.robot.IKTargetDoMove(
            position=reference_gripper_pos,
            duration=1,
            speed_based=False,
        )
        self.robot.WaitDo()
        self.env.step(50)
          
          
    
    def approach_manikin(self):
         # Turn to face Manikin
        robot.TurnLeft(90, 1)
        env.step(300)

        # Move towards Manikin
        robot.MoveForward(0.90, 2)
        env.step(200)
        
        robot.TurnLeft(90, 1)
        env.step(300)
        
        # Move towards Manikin
        robot.MoveBack(0.20, 2)
        env.step(150)
        
        # print(f"robot position: {robot.data.get('position')}")
        # print(f"robot rotation: {robot.data.get('rotation')}")
        
        
        
    def scrub_manikin(self):
        # PD Controller gains - Robot arm
        k_p = 1.0
        k_d = 2.0 * np.sqrt(k_p)

        # Admittance Controller parameters - Manikin
        mass       = 1000        # virtual mass (kg·m²)
        stiffness  = 1      # virtual spring constant (N/m)
        damping    = 2.0 * np.sqrt(stiffness * mass)  # critically damped
        desired_force = 6.0       # N

        # Discretisation time step
        dt = 1  # in seconds

        # indices = [4, 5, 6, 7]
        indices = [6, 7]
        # get the EE's current and target positions
        current = self.gripper.data.get("grasp_point_position")
        print(f"Initial:  {current}")
        
        start = self.trajectories[indices[0]]
        target = self.trajectories[indices[1]]
        print(f"Initial Target: {target}")
        error = np.array(start) - np.array(current)
        print(f"error: {error}")
        
         # align the robot x axis (base)
        timeout = 10
        start_time = time.time()
        while abs(error[0]) > 0.05 and (time.time() - start_time) < timeout:
            if error[0] < 0:
                self.robot.MoveBack(-error[0], 1)
                self.env.step(100)
            else:
                self.robot.MoveForward(error[0], 1)
                self.env.step(100)
            error = np.array(start) - np.array(self.gripper.data.get("grasp_point_position"))
        elapsed = time.time() - start_time
        print(f"Took {elapsed:.2f}s (timeout was {timeout}s)")
        print(f"Error after x correction: {error}")
             
        # align the z axis - define a timeout (in seconds)
        timeout = 30
        start_time = time.time()
        while abs(error[2]) > 0.05 and (time.time() - start_time) < timeout:
            self.robot.IKTargetDoMove(
                position=[0, 0, error[2]],
                duration=4,
                speed_based=False,
                relative=True
            )
            self.robot.WaitDo()
            self.env.step(100)
            error = np.array(start) - np.array(self.gripper.data.get("grasp_point_position"))
        elapsed = time.time() - start_time
        print(f"Error after z correction: {error}")
        print(f"Took {elapsed:.2f}s (timeout was {timeout}s)")
        
        # self.robot.SetImmovable(True)
        print(f"Final pos: {self.gripper.data.get('grasp_point_position')}")
        print(f"Desired pos: {self.trajectories[indices[0]]}")
        
        # Probe
        contact = False
        px = 0
        py = 0
        pz = 0
        
        expert_obs = []
        expert_actions = []
        goals = []
        
        
        while contact == False:
            # get the current position and force
            current = np.array(self.gripper.data.get('grasp_point_position'))
            ee = np.array(self.gripper.data.get('grasp_point_position'))
            force = float(self.sponge.GetForce()[0])
            pm = self.person_midpoint
            user_input = int(input("0 to move downward; 1 to move forward; 2 to move upward; 3 to move backward; 4 to move robot base forward; 5 to move robot base backward; 6 to end:"))
            
            
            # expert_obs
            if user_input == 0:
                expert_obs.append([ee[0], ee[1], ee[2], force, pm[0], pm[1], pm[2]])
                # add a small y-offset 
                dxyz = np.array([0.0, -0.02, 0.0])
                end_ee = current + dxyz
                self.robot.IKTargetDoMove(
                    position=end_ee.tolist(),
                    duration=2,
                    speed_based=False,
                    relative=False
                )
                self.robot.WaitDo()
                self.env.step(100)
                py += -0.02
                ee = self.gripper.data.get('grasp_point_position')
                force = float(self.sponge.GetForce()[0])
                pm = self.person_midpoint
                print(f"self.person_midpoint: {self.person_midpoint}")
                expert_actions.append(0)
                goals.append("leg")
                self.env.step()
            elif user_input == 1:
                expert_obs.append([ee[0], ee[1], ee[2], force, pm[0], pm[1], pm[2]])
                # add a small z-offset 
                dxyz = np.array([0, 0.0, -0.05])
                end_ee = current + dxyz
                self.robot.IKTargetDoMove(
                    position=end_ee.tolist(),
                    duration=2,
                    speed_based=False,
                    relative=False
                )
                # self.robot.IKTargetDoComplete()
                self.robot.WaitDo()
                self.env.step(100)
                pz += -0.05
                ee = self.gripper.data.get('grasp_point_position')
                force = float(self.sponge.GetForce()[0])
                pm = self.person_midpoint
                print(f"self.person_midpoint: {self.person_midpoint}")
                expert_actions.append(1)
                goals.append("leg")
                self.env.step()
            elif user_input == 2:
                expert_obs.append([ee[0], ee[1], ee[2], force, pm[0], pm[1], pm[2]])
                # add a small y-offset 
                dxyz = np.array([0.0, 0.02, 0.0])
                end_ee = current + dxyz
                self.robot.IKTargetDoMove(
                    position=end_ee.tolist(),
                    duration=2,
                    speed_based=False,
                    relative=False
                )
                self.robot.WaitDo()
                self.env.step(100)
                py += 0.02
                ee = self.gripper.data.get('grasp_point_position')
                force = float(self.sponge.GetForce()[0])
                pm = self.person_midpoint
                print(f"self.person_midpoint: {self.person_midpoint}")
                expert_actions.append(0)
                goals.append("leg")
                self.env.step()
            elif user_input == 3:
                # add a small z-offset 
                dxyz = np.array([0, 0.0, 0.05])
                end_ee = current + dxyz
                self.robot.IKTargetDoMove(
                    position=end_ee.tolist(),
                    duration=2,
                    speed_based=False,
                    relative=False
                )
                # self.robot.IKTargetDoComplete()
                self.robot.WaitDo()
                self.env.step(100)
                pz += 0.05
            elif user_input == 4:
                self.robot.MoveForward(0.1, 1)
                self.env.step(100)
            elif user_input == 5:
                self.robot.MoveBack(0.1, 1)
                self.env.step(100)
            elif user_input == 6:
                obj = self.env.GetAttr(666666)
                self.env.step()
                print(obj.data)
            else:
                print("invalid input. choose [0-6]")
            # end if user input is 2 or contact force > 6N
            force = self.sponge.GetForce()[0]
            if user_input == 6:
                contact = True
            # check if contact is made
            if force >= 3:
                contact = True
            print(f"Final pos: {self.gripper.data.get('grasp_point_position')}")
            print(f"shift: {px, py, pz}")
          
        expert_data = {
            "obs": expert_obs,
            "acts": expert_actions,
            "goal": goals
        }  
        with jsonlines.open('expert_traj_action.jsonl', mode='a') as writer:
            writer.write(expert_data)
        print(f"expert data ***{expert_data} *** saved")
            
        # 1.    Now, add this probe differential to the desired trajectory
        # Prevent drilling effect:  
        py = max(py, -0.14)
        target = [target[0] + px, target[1] + py, target[2] + pz]
        start = [start[0] + px, start[1] + py, start[2] + pz]
        print(f"Final pos: {target}")
        
        distance = target[0] - current[0]
        N_steps = 10

        # 2.    Discretise the trajectory into N steps
        # PD Controller
        prev_error = [0, 0, 0]
        error = [0, 0, 0]
        dt = 1
        current = np.array(self.robot.data.get("position"))
        while abs(self.robot.data.get("position")[0] - target[0]) > 0.01:
            self.robot.SetImmovable(True)
            # get progress ratio
            ratio = abs(self.robot.data.get("position")[0] - current[0]) / abs(target[0] - current[0])
            # compute desired ee position at this point
            desired_ee = [current[0] + ratio * (target[0] - current[0]) + 0.3, current[1], current[2] + ratio * (target[2] - current[2]) + 0.3]
            current_ee = self.robot.data.get("position")
            error = [0, 0, desired_ee[2] - current_ee[2]]
            # e_derivative = [(error[0] - prev_error[0]) / dt, (error[1] - prev_error[1]) / dt, (error[2] - prev_error[2]) / dt]
            # u_t = [(k_p * error[0]) + (k_d * e_derivative[0]), 0, (k_p * error[2]) + (k_d * e_derivative[2])]
            # new_ee = [current_ee[0] + u_t[0], current_ee[1] + u_t[1], current_ee[2] + u_t[2]]
            # command the robot to move to this position
            # new_ee = [desired_ee[0] + error[0], desired_ee[1] + error[1], desired_ee[2] + error[2]]
            self.robot.IKTargetDoMove(
                position=[current[0] + 0.15, 0, 0],
                duration=5,
                speed_based=False,
                relative=True
            )
            # self.robot.IKTargetDoComplete()
            self.robot.WaitDo()
            self.robot.IKTargetDoComplete()
            self.env.step(200)
            prev_error = error
            # self.robot.MoveForward(distance/N_steps, 0.5)
            
            
        
        # 3.    Implement Admittance Control with PD controller

        
        
    def run(self):
        self.grasp_sponge()
        self.dip_sponge()
        self.approach_manikin()
        self.scrub_manikin()
        
        
      

class Perception:
    
    # Setup a camera to capture objects in the scene
    def __init__(self, env:BathingEnv):
        # Get an instance of the environment
        self.env = env  
        
        # Setup a camera to capture the entire scene
        self.scene_camera = env.InstanceObject(name="Camera", id=123456, attr_type=attr.CameraAttr) 
        # Set position and orientation for the scene camera
        self.scene_camera.SetTransform(position=[0, 4.2, 0.6], rotation=[90, 0, 0.0])   
        
        # Setup a camera to get manikin data
        self.manikin_camera = env.InstanceObject(name="Camera", id=333333, attr_type=attr.CameraAttr)
        self.manikin_camera.SetTransform(position=[0.0, 2.1, 0.15], rotation=[90, 0, 0])
        
        # 4) camera for LLM obs
        self.obscamera = self.env.InstanceObject(name="Camera", id=666666, attr_type=attr.CameraAttr)
        # self.obscamera.SetTransform(position=[-0.1, 2, -0.1], rotation=[60, 45, 0.0])
        # self.obscamera.SetTransform(position=[-0.4, 1.36, 0.36], rotation=[40, 105, 0])
        self.obscamera.SetTransform(position=[1, 2.0, 0.0], rotation=[45, 270, 0])

        self.env.step()
        
        self.landmarks_dict = {}
        self.l2w_poses= {}
        self.trajectories = []
        self.landmarks = None
        self.env.step(50)
        
        # Update landmark information
        self.save_landmarks()
    
    # Get sponge 
    def get_sponge(self):
        sponge = self.env.get_sponge()
        return sponge
    
    # Get robot directly from the environment
    def get_robot(self):
        robot = self.env.get_robot()
        return robot
 
    # Get gripper 
    def get_gripper(self):
        gripper = self.env.get_gripper()
        return gripper
    
    # Get robot (id = 221582) from camera
    def get_robot_from_camera(self):
        coords = self._get_coords(221582)
        return coords
    
    def convert_to_homcoords(self, landmarks):
        # swap y and z values to match RCareworld convention
        # Append 1 to each values array to convert to homogeneous coords
        for key, value in landmarks.items():
            self.landmarks_dict[key] = [value.x, value.y, value.z, 1]
    
    def convert_to_world_coords(self, l2wm):
        for key, point in self.landmarks_dict.items():
            # Convert the point to a numpy array
            local_point = np.array(point)
            # Multiply by the transformation matrix 
            world_point = l2wm @ local_point
            # Normalise by the w component (if not 1)
            world_point /= world_point[3]
            # Store the resulting x, y, z 
            # TODO: Placeholder for correct depth
            # print(f"World point before: {world_point}")
            if key == "right_wrist" or key == "left_wrist":
                world_point[1] = 0.93
            elif key == "right_shoulder" or key == "left_shoulder":
                world_point[1] = 0.96
            elif key == "right_hip" or key == "left_hip":
                world_point[1] = 0.96
            elif key == "right_ankle" or key == "left_ankle":
                world_point[1] = 0.95
            # world_point[1] = 0.93
            # print(f"World point after: {world_point}")
            self.l2w_poses[key] = world_point[:3]
            
    def save_landmarks(self):
        # ------------------ Scrub manikin perception -----------------------
        self.env.step(50) 
        # Capture an image of the manikin from the manikin camera
        image = self.manikin_camera.GetRGB(width=512, height=512)
        self.env.step()
        self.rgb = np.frombuffer(self.manikin_camera.data["rgb"], dtype=np.uint8)
        self.rgb = cv2.imdecode(self.rgb, cv2.IMREAD_COLOR)
        if self.rgb is None:
            print("Error: Decoded image is None.")
            return
        else:
            print("Image successfully captured. Image shape:", self.rgb.shape)  
        
        # image = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2BGR)
        # cv2.imshow("Manikin Camera View", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        # Decode image and check validity
        
        # # Detect pose
        detector = DetectPose(self.rgb)  # setup a detector
        new, lmarks = detector.get_landmarks()  # get landmarks from the detector + check if there are new landmarks
        # print(lmarks)
        
        if not new:
            print("No detection...") 
            self.env.close()
        
        # convert lmarks to homogeneous coordinates
        self.convert_to_homcoords(lmarks)
        
        # Get the camera's local to world matrix
        local_to_world_matrix = self.manikin_camera.data.get("local_to_world_matrix")
        
        # convert to world coordinates
        self.convert_to_world_coords(local_to_world_matrix)
        
        # Save the l2w poses to a json file for further investigation
        converted_poses = {key: value.tolist() for key, value in self.l2w_poses.items()}
        # print(f"Converted landmarks: {converted_poses}")
        with open("landmarks.json", "w") as file:
            json.dump(converted_poses, file, indent=4)
            print("Landmarks saved to landmarks.json")
            
            
    def get_landmark(self, key="right_shoulder"):
        # Read in the landmarks from the landmarks.json file
        try:
            with open('landmarks.json', 'r') as f:
                self.landmarks = json.load(f)
        except Exception as e:
            print("Error reading landmarks.json:", e)
            # self.landmarks = {}
        return self.landmarks[key]
    
    
    def get_landmarks(self):
        try:
            with open('landmarks.json', 'r') as f:
                self.landmarks = json.load(f)
        except Exception as e:
            print("Error reading landmarks.json:", e)
        return self.landmarks
        
    # position': [-0.11999999731779099, 1.3600000143051147, 0.30000001192092896], 'rotation': [40.00000762939453, 105.0, -2.2290444121608743e-06]
    
    def generate_trajectory(self):        
        try:
            # Extract the necessary landmarks
            right_shoulder = self.landmarks["right_shoulder"]
            right_wrist    = self.landmarks["right_wrist"]
            left_shoulder  = self.landmarks["left_shoulder"]
            left_wrist     = self.landmarks["left_wrist"]
            right_hip      = self.landmarks["right_hip"]
            right_ankle = self.landmarks["right_ankle"]
            left_hip       = self.landmarks["left_hip"]
            left_ankle  = self.landmarks["left_ankle"]
        except KeyError as e:
            raise Exception(f"Missing landmark in the data: {e}")
        
        self.trajectories.append(right_wrist)
        self.trajectories.append(right_shoulder)
        self.trajectories.append(right_hip)
        self.trajectories.append(right_ankle)
        self.trajectories.append(left_wrist)
        self.trajectories.append(left_shoulder)
        self.trajectories.append(left_hip)
        self.trajectories.append(left_ankle)
        
        # # TODO: Place holder for depth error correction
        # if self.depth_error is not None:
        #     return self.trajectories


if __name__ == "__main__":

    # Initialise the environment
    env = BathingEnv(graphics=True)  
    # Create a perception object. Detected poses should now be available
    p = Perception(env)
    # env.AlignCamera(123456)  # Scene camera
    # env.AlignCamera(333333) # Mannequin camera
    env.AlignCamera(666666) # Observation camera
    robot = p.get_robot()
    sponge = p.get_sponge()
    gripper = p.get_gripper()
    
    # ------------------- Trajectory Generation -----------------------------
    # Initialise a trajectory generator to get the trajectories
    p.get_landmarks()  #  first load the landmarks
    p.generate_trajectory()  # generate the trajectories
    
    task_runner = ReachMannequin(env, robot, gripper, sponge, p.trajectories)
    task_runner.run()
    
    # Do not close window
    env.WaitLoadDone()
  
  
        
        


# class Prober:    
#     def __init__(self, env:BathingEnv):
#         pass