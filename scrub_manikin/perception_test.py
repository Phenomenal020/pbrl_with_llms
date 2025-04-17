# Perception class:
# To get the locations of objects in the scene.
# This file was mostly used during the initial stages to know the locations of objects. Subsequently, the code works fine without it.

from pyrcareworld.envs.base_env import RCareWorld
from pyrcareworld.envs.bathing_env import BathingEnv
import pyrcareworld.attributes.camera_attr as attr
import numpy as np
import cv2

import mediapipe as mp 
import json

from detect_pose import DetectPose

from generate_trajectories import GenTrajectory

# import pyrcareworld.attributes as _attr
from pyrcareworld.attributes.omplmanager_attr import OmplManagerAttr

# PDController 
# from pid import PDController

try:
    import pyrcareworld.attributes.omplmanager_attr as rfu_ompl
except ImportError:
    raise Exception("This feature requires ompl, see: https://github.com/ompl/ompl")


class Perception:
    
    # Setup a camera to capture objects in the scene
    def __init__(self, env:BathingEnv):
        # Get an instance of the environment
        # self.ompl_manager = env.InstanceObject(name="OmplManager", attr_type=OmplManagerAttr)
        self.env = env  
        
        # Setup a camera to capture the entire scene
        self.scene_camera = env.InstanceObject(name="Camera", id=123456, attr_type=attr.CameraAttr) 
        # Set position and orientation for the scene camera
        self.scene_camera.SetTransform(position=[0, 4.2, 0.6], rotation=[90, 0, 0.0])   
        
        # Setup a camera to get manikin data
        self.manikin_camera = env.InstanceObject(name="Camera", id=333333, attr_type=attr.CameraAttr)
        self.manikin_camera.SetTransform(position=[0.0, 2.1, 0.15], rotation=[90, 0, 0])
        
        self.test_camera = env.InstanceObject(name="Camera", id=444444, attr_type=attr.CameraAttr)
        self.test_camera2 = env.InstanceObject(name="Camera", id=555555, attr_type=attr.CameraAttr)
        self.test_camera3 = env.InstanceObject(name="Camera", id=666666, attr_type=attr.CameraAttr)
        self.test_camera4 = env.InstanceObject(name="Camera", id=777777, attr_type=attr.CameraAttr)
        self.test_camera5 = env.InstanceObject(name="Camera", id=888888, attr_type=attr.CameraAttr)
        self.test_camera6 = env.InstanceObject(name="Camera", id=999999, attr_type=attr.CameraAttr)
        self.test_camera7 = env.InstanceObject(name="Camera", id=101010, attr_type=attr.CameraAttr)
        self.test_camera8 = env.InstanceObject(name="Camera", id=111111, attr_type=attr.CameraAttr)
        
        self.landmarks_dict = {}
        self.l2w_poses= {}
        self.trajectories = []
        self.landmarks = None
        self.env.step(50)
        
        # Update landmark information
        self.save_landmarks()

        
        # right shoulder - ✅
        self.test_camera.SetTransform(position=self.get_landmark("right_shoulder"), rotation=[90, 0, 0])
        
        # right wrist - ✅
        self.test_camera2.SetTransform(position=self.get_landmark("right_wrist"), rotation=[90, 0, 0])
        
        # left shoulder - ✅
        self.test_camera3.SetTransform(position=self.get_landmark("left_shoulder"), rotation=[90, 0, 0])
        
        # left wrist - ✅
        self.test_camera4.SetTransform(position=self.get_landmark("left_wrist"), rotation=[90, 0, 0])

        # right hip - ✅
        self.test_camera5.SetTransform(position=self.get_landmark("right_hip"), rotation=[90, 0, 0])
        
        # left hip - ✅
        self.test_camera6.SetTransform(position=self.get_landmark("left_hip"), rotation=[90, 0, 0])
        
        # right ankle - ✅
        self.test_camera7.SetTransform(position=self.get_landmark("right_ankle"), rotation=[90, 0, 0])
        
        # left ankle - ✅
        self.test_camera8.SetTransform(position=self.get_landmark("left_ankle"), rotation=[90, 0, 0])
        
    
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
                world_point[1] = 0.80
            elif key == "right_shoulder" or key == "left_shoulder":
                world_point[1] = 0.80
            elif key == "right_hip" or key == "left_hip":
                world_point[1] = 0.80
            elif key == "right_ankle" or key == "left_ankle":
                world_point[1] = 0.80
            # world_point[1] = 0.93
            # print(f"World point after: {world_point}")
            self.l2w_poses[key] = world_point[:3]
            
    def save_landmarks(self):
        # --------------------------------------------------------------- Scrub manikin perception task --------------------------------------------------------------------
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
    env.AlignCamera(333333) # Mannequin camera
    
    robot = p.get_robot()
    sponge = p.get_sponge()
    gripper = p.get_gripper()

    
    
    # ----------------------------------------------- Trajectory Generation ----------------------------------------------------
    # Initialise a trajectory generator to get the trajectories
    p.get_landmarks()  #  first load the landmarks
    p.generate_trajectory()  # generate the trajectories
    # print(f"Landmarks: {p.landmarks}")
    # print(f"Trajectories: {p.trajectories}")
    
    
    # ----------------------------------------------- Trajectory execution -----------------------------------------------------
    for traj in p.trajectories:
        pass
    

    # ----------------------------------------------- Grasp sponge and Dip water tank -------------------------------------------
    # Raise the gripper to a safe height for easy obstacle avoidance
    gripper_pos = [robot.data["position"][0], sponge.data.get("position")[1] + 0.3, robot.data["position"][2]]
    print(f"Gripper position: {gripper_pos}")
    robot.IKTargetDoMove(
        position=gripper_pos,
        duration=1,
        speed_based=False,
    )
    robot.WaitDo()
    env.step(50)
    gripper.GripperOpen()
    env.step(50)
    
    # Move towards sponge
    robot.MoveForward(0.25, 2)
    env.step(100)
    
    # Turn left
    robot.TurnLeft(90, 1.5)
    env.step(200)
    
    # Move forward 
    robot.MoveForward(0.2, 2)
    env.step(50)
    
    # Lower gripper (Grasp sponge)
    lower_position = [sponge.data.get("position")[0], sponge.data.get("position")[1]+0.03, sponge.data.get("position")[2]]
    robot.IKTargetDoMove(
        position=lower_position,  
        duration=1,
        speed_based=False,
    )
    robot.WaitDo()
    env.step(50)
    
    # Grasp sponge
    gripper.GripperClose()
    env.step(50)
    
    # Raise Gripper
    robot.IKTargetDoMove(
        position=gripper_pos,
        duration=1,
        speed_based=False,
    )
    robot.WaitDo()
    env.step(50)
    
    # Move backwards
    robot.MoveBack(0.40, 2)
    env.step(50)
    
    print(gripper.data)
    
    # Lower Gripper to dip sponge
    robot.IKTargetDoMove(
        position=lower_position,  
        duration=1,
        speed_based=False,
    )
    robot.WaitDo()
    env.step(50)
    
    # Raise Gripper
    reference_gripper_pos = [robot.data["position"][0], sponge.data.get("position")[1] + 0.4, robot.data["position"][2]]
    robot.IKTargetDoMove(
        position=reference_gripper_pos,
        duration=1,
        speed_based=False,
    )
    robot.WaitDo()
    env.step(50)
    
    # Turn to face Manikin
    robot.TurnRight(270, 1)
    env.step(600)
    
    # Move towards Manikin
    robot.MoveForward(0.9, 2)
    env.step(150)
    
    robot.TurnLeft(90, 1)
    env.step(300)
    
    start_indices = [4, 5, 6]    # left wrist, left shoulder, left hip
    target_indices = [5, 6, 7]  # left shoulder, left hip, left ankle
    
    for i in range(len(start_indices)):
        start = p.trajectories[start_indices[i]] 
        x_start = start[0]
        y_start = start[1]
        z_start = start[2]
        
        target = p.trajectories[target_indices[i]]  
        x_target = target[0]
        y_target = target[1]
        z_target = target[2]
    
        x_steps = (x_target - x_start) / 5
        y_steps = (y_target - y_start) / 5
        z_steps = (z_target - z_start) / 5
        
        desired_start = start
        
        print(f"Start: {start}")
        print(f"Target: {target}")
        print(f"Steps: {x_steps, y_steps, z_steps}")
                    
        for j in range(1, 6):
            
            print(j)

            # At each loopstep, calculate the desired position
            desired_pos = np.array([desired_start[0] + (x_steps*j), desired_start[1] + (y_steps*j), desired_start[2] + (z_steps*j)])
            print(f"Desired position: {desired_pos}")
            # Move to desired position
            robot.SetImmovable(True)
            robot.IKTargetDoMove(
                position=desired_pos.tolist(),  
                duration=5.0,
                speed_based=False,
                relative=False
            )  
            robot.IKTargetDoComplete()
            robot.WaitDo()
            env.step(300)
            curr_pos = gripper.data.get("grasp_point_position")  # Get current EE position
            error = np.array([desired_pos[0] - curr_pos[0], desired_pos[1] - curr_pos[1], desired_pos[2] - curr_pos[2]])  # Calculate difference from desired position
            print(f"Initial error: {np.linalg.norm(error)}")  # Print initial difference
            robot.IKTargetDoMove(
                position=error.tolist(),  
                duration=5.0,
                speed_based=False,
                relative=True
            )  
            robot.IKTargetDoComplete()
            robot.WaitDo()
            env.step(300)
            curr_pos = gripper.data.get("grasp_point_position") 
            new_diff = np.array([desired_pos[0] - curr_pos[0], desired_pos[1] - curr_pos[1], desired_pos[2] - curr_pos[2]])
            raise_diff_norm = np.linalg.norm(new_diff)
            print(f"New error: {raise_diff_norm}")  
             
            robot.SetImmovable(False)   
            # Move the robot to scrub. TODO: Admittance control if the force exceeds 6N
            if x_steps >= 0:
                robot.MoveForward(x_steps, 0.1) 
                env.step(300)
                # curr_pos = robot.data.get("position")
                # robot_x_diff = curr_pos[0] - desired_robot_posx
                # print(f"robot x_diff: {robot_x_diff}") 
                # robot_x_diff = np.linalg.norm(robot_x_diff)    
            else:
                #  robot_x_diff = 1.0
                #  while robot_x_diff > 0.1:
                #     desired_robot_posx = robot.data.get("position")[0] + x_steps
                robot.MoveBack(-x_steps, 0.1) 
                env.step(300)
                # curr_pos = robot.data.get("position")
                # robot_x_diff = curr_pos[0] - desired_robot_posx
                # print(f"robot x_diff: {robot_x_diff}") 
                # robot_x_diff = np.linalg.norm(robot_x_diff)   
            
  
    # Do not close window
    env.WaitLoadDone()
    
    # self.data[‘result_joint_position’]
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
    # PD Controller gains and parameters.
    # Kp = np.array([0.1, 0, 0.1])     
    # Kd = 2 * np.sqrt(Kp)     
    # dt = 1.0   
    # previous_error = np.array([0, 0, 0])
    # u_t = np.array([0, 0, 0])
    
      
    # # # -------------------------------------- OMPL Integration -----------------------------
    # # robot.EnabledNativeIK(False) # native inverse kinematics (IK) is disabled
    # # env.step()
    
    # # # p.ompl_manager = env.InstanceObject(name="OmplManager", attr_type=_attr.OmplManagerAttr)
    # # p.ompl_manager.modify_robot(221582)  # o that it can plan motions for that robot.
    # # env.step()
    
    # # planner = rfu_ompl.RFUOMPL(p.ompl_manager, time_unit=5)

    # # print("+++++++++++++++ OMPL Working +++++++++++++++")
    
    # # start_state_cs = [
    # #     -0.020131396129727364,
    # #     2.3013671402469313,
    # #     -0.2289543200973192
    # # ]
    # # robot.GetIKTargetJointPosition(start_state_cs, iterate=100)
    # # env.step()
    # # start_state_js =  robot.data['result_joint_position']
    # # print(f"++++++ {start_state_js} +++++++ ")
    
    # # target_state_cs = [
    # #     0.4879406988620758,
    # #     2.165602606022067,
    # #     0.018232385705884813
    # # ]
    # # robot.GetIKTargetJointPosition(target_state_cs, iterate=100)
    # # env.step()
    # # target_state_js =  robot.data["result_joint_position"]
    # # print(f"++++++ {target_state_js} +++++++ ")

    # # # begin
    # # p.ompl_manager.set_state(start_state_js)
    # # env.step(50)

    # # # target
    # # p.ompl_manager.set_state(target_state_js)
    # # env.step(50)

    # # # return
    # # p.ompl_manager.set_state(start_state_js)
    # # env.step(50)

    # # # The simulation’s time step is set to a very small value (0.001) and then stepped to register this change.
    # # env.SetTimeStep(0.001)
    # # env.step()

    # # # is_sol is a boolean flag indicating whether a valid solution (path) was found.
    # # # path is the sequence of states (or waypoints) that the planner computed.
    # # is_sol, path = planner.plan_start_goal(target_state_js, target_state_js)

    # # # The code prints the target state and the last state in the computed path. This is likely to verify that the planned path ends at the desired target configuration.
    # # print(target_state_js)
    # # print(path[-1])

    # # # The time step is increased back to 0.02 for executing the path at a normal simulation pace.
    # # env.SetTimeStep(0.02)
    # # env.step()


    # # # if a valid solution was found (is_sol is True), it continuously executes the planned path.
    # # while True:
    # #     if is_sol:
    # #         planner.execute(path)
    