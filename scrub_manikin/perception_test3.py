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

from detect_pose import DetectPose

from generate_trajectories import GenTrajectory

# import pyrcareworld.attributes as _attr
from pyrcareworld.attributes.omplmanager_attr import OmplManagerAttr
from pyrcareworld.attributes import PersonRandomizerAttr

# Import necessary mediapipe libraries
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image, ImageFormat

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
        self.manikin_indices = [4, 5, 6, 7]
        self.target = None
        self.start = None
        
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
        reference_gripper_pos = [self.robot.data["position"][0], self.initial_sponge_pos[1] + 0.4, self.robot.data["position"][2]]
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
        robot.MoveBack(0.10, 2)
        env.step(150)
        
    
    def probe(self):
        for i in range(0, len(self.manikin_indices)-1):
            # get the EE's current and target positions
            current = self.gripper.data.get("grasp_point_position")
            
            start = self.trajectories[self.manikin_indices[i]]
            target = self.trajectories[self.manikin_indices[i+1]]
            print(f"Target before probe: {target}")
            
            error = np.array(start) - np.array(current)
            print(f"Error before correction: {error}")
            max_steps = 0
            
            # align the x axis
            while abs(error[0]) > 0.05 and max_steps <5000:
                if error[0] < 0:
                    self.robot.MoveBack(-error[0], 1)
                    max_steps += 100
                    self.env.step(100)
                else:
                    self.robot.MoveForward(error[0], 1)
                    max_steps += 100
                    self.env.step(100)
                error = np.array(start) - np.array(self.gripper.data.get("grasp_point_position"))
                
            print(f"Error after x correction: {error}")
            
            
            # align the y axis
            while abs(error[1]) > 0.05 and max_steps <5000:
                self.robot.IKTargetDoMove(
                    position=[0, error[1], 0],
                    duration=5,
                    speed_based=False,
                    relative=True
                )
                self.robot.WaitDo()
                max_steps += 100
                self.env.step(100)
                error = np.array(start) - np.array(self.gripper.data.get("grasp_point_position"))
            print(f"Error after y correction: {error}")
        
            # align the z axis
            while abs(error[2]) > 0.05 and max_steps <5000:
                self.robot.IKTargetDoMove(
                    position=[0, 0, error[2]],
                    duration=5,
                    speed_based=False,
                    relative=True
                )
                self.robot.WaitDo()
                max_steps += 100
                self.env.step(100)
                error = np.array(start) - np.array(self.gripper.data.get("grasp_point_position"))
            
            print(f"Error after z correction: {error}")
                
            # Probe
            contact = False
            px = 0
            py = 0
            pz = 0
            
            while contact == False:
                # get the current position and force
                current = np.array(self.gripper.data.get('grasp_point_position'))
                user_input = int(input("0 to move downward; 1 to move forward; 2 to move upward; 3 to move backward; 4 to move robot base forward; 5 to move robot base backward; 6 to end:"))
            
                if user_input == 0:
                    # move downward
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
                    
                elif user_input == 1:
                    # move forward
                    dxyz = np.array([0, 0.0, -0.05])
                    end_ee = current + dxyz
                    self.robot.IKTargetDoMove(
                        position=end_ee.tolist(),
                        duration=2,
                        speed_based=False,
                        relative=False
                    )
                    self.robot.WaitDo()
                    self.env.step(100)
                    pz += -0.05
                
                elif user_input == 2:
                    # move upward
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
                    
                elif user_input == 3:
                    # move backward
                    dxyz = np.array([0, 0.0, 0.05])
                    end_ee = current + dxyz
                    self.robot.IKTargetDoMove(
                        position=end_ee.tolist(),
                        duration=2,
                        speed_based=False,
                        relative=False
                    )
                    self.robot.WaitDo()
                    self.env.step(100)
                    pz += 0.05
                    
                elif user_input == 4:
                    # move robot base forward
                    self.robot.MoveForward(0.1, 1)
                    self.env.step(100)
                    px += 0.1
                    
                elif user_input == 5:
                    # move robot base backward
                    self.robot.MoveBack(0.1, 1)
                    self.env.step(100)
                    px += -0.1
                
                else:
                    print("invalid input. choose [0-6]")
                    
                # end if user input is 6 or contact force > 3N
                force = self.sponge.GetForce()[0]
                if user_input == 6:
                    contact = True
                # if force >= 5:
                #     contact = True
            
            print(f"shift: {px, py, pz}")
            
            # Add this probe differential to the desired trajectory
            # Prevent drilling effect:
            py = max(py, -0.12)
            print(f"Start before probing: {start}")
            print(f"Target before probing: {target}")
            target = [target[0] + px, target[1] + py, target[2] + pz]
            start = [start[0] + px, start[1] + py, start[2] + pz]
            print(f"Start after probing: {start}")
            print(f"Target after probing: {target}")
            self.start = start
            self.target = target
            self.scrub_manikin(target, start)
            # End probing
        
        
        
    def scrub_manikin(self, target, start):
        # target = self.target
        # start = self.start
        # PD Controller gains 
        k_p = 0.01
        k_d = 2.0 * np.sqrt(k_p)
        prev_error = [0, 0, 0]
        error = [0, 0, 0]
        dt = 1
        initial = np.array(self.gripper.data.get("grasp_point_position"))

        # Admittance Controller parameters - Manikin
        mass       = 1000        # virtual mass (kg·m²)
        stiffness  = 1      # virtual spring constant (N/m)
        damping    = 2.0 * np.sqrt(stiffness * mass)  # critically damped
        desired_force = 6.0       # N
        dt = 1  # in seconds
           
        # Discretise the trajectory into N steps
        distanceX = target[0] - initial[0]
        # distanceY = target[1] - initial[1]
        distanceZ = target[2] - initial[2]

        N_steps = 5
        
        xstep = distanceX / N_steps
        # ystep = distanceY / N_steps
        zstep = distanceZ / N_steps 
        print(f"xstep ---> {xstep}, ***** zstep ---> {zstep}")
        
        # While robot not at target:
        for i in range(N_steps):
            
            # Fix robot base
            self.robot.SetImmovable(True)
            
            # compute desired ee position at this point
            desired_ee = [initial[0] + xstep*i, initial[1], initial[2] + zstep*i]             
            # determine the error at this step
            prev_ee = current_ee = self.gripper.data.get("grasp_point_position")
            error = [desired_ee[0] - current_ee[0], desired_ee[1] - current_ee[1], desired_ee[2] - current_ee[2]]
            print(f"Initial error: {error}")
            
            # e_derivative = [(error[0] - prev_error[0]) / dt, (error[1] - prev_error[1]) / dt, (error[2] - prev_error[2]) / dt]
            # u_t = [(k_p * error[0]) + (k_d * e_derivative[0]), (k_p * error[1]) + (k_d * e_derivative[1]), (k_p * error[2]) + (k_d * e_derivative[2])]
            # print(f"u_t: {u_t}")
            # new_ee = [current_ee[0] + u_t[0], current_ee[1] + u_t[1] - 0.02, current_ee[2] + u_t[2]]
            
            # Cover the error
            self.robot.IKTargetDoMove(
                position=desired_ee,
                duration=5,
                speed_based=False,
                relative=False
            )
            self.robot.WaitDo()
            self.env.step(200)
            print(f"desired ee: {desired_ee} || current ee: {self.gripper.data.get('grasp_point_position')}")
            
            # Sweep to new position
            self.robot.IKTargetDoMove(
                position=[3.0*xstep, -0.05, 3.0*zstep],
                duration=5,
                speed_based=False,
                relative=True
            )
            self.robot.WaitDo()
            self.env.step(200)
            print(f"desired ee: {desired_ee} || current ee: {self.gripper.data.get('grasp_point_position')}")

            # Return to prev pos
            self.robot.IKTargetDoMove(
                position=[3.0*-xstep, 0.05, 3.0*-zstep],
                duration=5,
                speed_based=False,
                relative=True
            )
            self.robot.WaitDo()
            self.env.step(200)
            print(f"desired ee: {desired_ee} || current ee: {self.gripper.data.get('grasp_point_position')}")
                        
            prev_error = error
            
            # Now move robot base
            self.robot.SetImmovable(False)
            
            if i == N_steps - 1:
                break
            
            # Raise gripper
            self.robot.IKTargetDoMove(
                position=[0, 0.3, 0],
                duration=5,
                speed_based=False,
                relative=True
            )
            self.robot.WaitDo()
            self.env.step(200)
            
            print(f"**********Raise gripper********** {self.gripper.data.get('grasp_point_position')}")
            
            if xstep > 0:
                self.robot.MoveForward(abs(xstep), 1)
            else:
                self.robot.MoveBack(abs(xstep), 1)
            self.env.step(100)
            print(f"**********Move base********** {self.gripper.data.get('grasp_point_position')}")
        
    def run(self):
        self.grasp_sponge()
        self.dip_sponge()
        self.approach_manikin()
        self.probe()
        # self.scrub_manikin()
        
        
      

class Perception:
    
    # Setup a camera to capture objects in the scene
    def __init__(self, env:BathingEnv, port):
        # Get an instance of the environment
        self.env = env  
        self.port = port
        
        self.landmarks_dict = {}
        self.l2w_poses= {}
        self.trajectories = []
        self.landmarks = None
        self.env.step()
        
        try:
            base_options = python.BaseOptions(
                model_asset_path=f"pose_landmarker{self.port}.task"
                )
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                output_segmentation_masks=False)
        except Exception as e:
            print(f"Landmarker error: {e}")

        self.image = None
        self.detector = vision.PoseLandmarker.create_from_options(options)
        
        # Update landmark information
        self.save_landmarks()
    
    # ******************************** NOT NEEDED  FOR RL INTEGRATION **************************************
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
    
    
    # ******************************* Utility functions **************************************
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
                world_point[1] = 0.88
            elif key == "right_shoulder" or key == "left_shoulder":
                world_point[1] = 0.94
            elif key == "right_hip" or key == "left_hip":
                world_point[1] = 0.94
            elif key == "right_ankle" or key == "left_ankle":
                world_point[1] = 0.94
            # world_point[1] = 0.93
            # print(f"World point after: {world_point}")
            self.l2w_poses[key] = world_point[:3]
            
    def save_landmarks(self):
        # ------------------ Scrub manikin perception -----------------------
        self.env.step(50) 
        # Capture an image of the manikin from the manikin camera
        image = self.env.manikin_camera.GetRGB(width=512, height=512)
        self.env.step()
        self.rgb = np.frombuffer(self.env.manikin_camera.data["rgb"], dtype=np.uint8)
        self.rgb = cv2.imdecode(self.rgb, cv2.IMREAD_COLOR)
        self.image = self.rgb
        # cv2.imwrite("blankrgb.png", self.rgb)
        # bgr = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2BGR)
        # cv2.imshow("Manikin Camera View", self.rgb)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        new, lmarks = self.detect_landmarks()  # get landmarks from the detector + 
        
        if not new:
            print("No new detection...") 
            return  # use old coordinates 
        
        # convert lmarks to homogeneous coordinates
        self.convert_to_homcoords(lmarks)
        # Get the camera's local to world matrix
        local_to_world_matrix = self.env.manikin_camera.data.get("local_to_world_matrix")
        # convert to world coordinates
        self.convert_to_world_coords(local_to_world_matrix)
        # Save the l2w poses to a json file for further investigation
        converted_poses = {key: value.tolist() for key, value in self.l2w_poses.items()}
        filename = f"landmarks{self.port}.json"
        # print(f"Converted landmarks: {converted_poses}")
        with open(filename, "w") as file:
            json.dump(converted_poses, file, indent=4)
            print(f"Landmarks saved to {filename}")
            
    def detect_landmarks(self):
        # Load the input image from a numpy array.
        mp_image = Image(image_format=ImageFormat.SRGB, data=self.image)
        # detect poses
        detection_result = self.detector.detect(mp_image)
        
        # print(f"Detection result: {detection_result}")

        # # # Process the detection result. In this case, visualise it.
        # annotated_image = self.draw_landmarks_on_image(mp_image.numpy_view(), detection_result)
        # cv2.imshow("Annotated_image", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Get landmarks in camera frame - not normalised
        try:
            world_landmarks = detection_result.pose_world_landmarks[0]
        except IndexError:
            print("No pose landmarks detected")
            return False, {}
            # world_landmarks = detection_result.pose_landmarks[0]
        
        
        # right arm
        right_wrist = world_landmarks[16]
        right_shoulder = world_landmarks[12]
        # left arm
        left_shoulder = world_landmarks[11]
        left_wrist = world_landmarks[15]
        # right leg
        right_hip = world_landmarks[24]
        right_ankle = world_landmarks[28]
        # left leg
        left_hip = world_landmarks[23]
        left_ankle = world_landmarks[27]
        # mid-body
        # mid_body = (right_shoulder, left_shoulder, left_hip, right_hip, right_shoulder)
        
        return True, {
        "right_wrist": right_wrist,
        "right_shoulder": right_shoulder,
        "left_shoulder": left_shoulder,
        "left_wrist": left_wrist,
        "right_hip": right_hip,
        "right_ankle": right_ankle,
        "left_hip": left_hip,
        "left_ankle": left_ankle,
        }
            
     
    # def draw_landmarks_on_image(self, rgb_image, detection_result):
    #     pose_landmarks_list = detection_result.pose_landmarks
    #     annotated_image = np.copy(rgb_image)

    #     # Loop through the detected poses to visualize.
    #     for idx in range(len(pose_landmarks_list)):
    #         pose_landmarks = pose_landmarks_list[idx]

    #         # Draw the pose landmarks.
    #         pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    #         pose_landmarks_proto.landmark.extend([
    #         landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    #         ])
    #         solutions.drawing_utils.draw_landmarks(
    #         annotated_image,
    #         pose_landmarks_proto,
    #         solutions.pose.POSE_CONNECTIONS,
    #         solutions.drawing_styles.get_default_pose_landmarks_style())
    #     return annotated_image
            
    def get_landmark(self, key="right_shoulder"):
        # Read in the landmarks from the landmarks.json file
        filename = f"landmarks{self.port}.json"
        try:
            with open(filename, 'r') as f:
                self.landmarks = json.load(f)
        except Exception as e:
            print(f"Error reading {filename}:", e)
            # self.landmarks = {}
        return self.landmarks[key]
    
    
    def get_landmarks(self):
        filename = f"landmarks{self.port}.json"
        try:
            with open(filename, 'r') as f:
                self.landmarks = json.load(f)
        except Exception as e:
            print(f"Error reading {filename}:", e)
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


if __name__ == "__main__":

    # Initialise the environment
    env = BathingEnv(graphics=True)  
    # Create a perception object. Detected poses should now be available
    p = Perception(env)
    env.AlignCamera(123456)  # Scene camera
    # env.AlignCamera(333333) # Mannequin camera
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
    
    
    

# Usage in main
if __name__ == '__main__':
    env = BathingEnv(graphics=True)
    p = Perception(env)
    env.AlignCamera(333333)
    robot, sponge, gripper = p.get_robot(), p.get_sponge(), p.get_gripper()
    p.get_landmarks(); 
    p.generate_trajectory()
    # define or load trained policy
    # def rl_policy(obs):
    #     # placeholder: random action
    #     return probe_env.action_space.sample()

    # reacher = ReachMannequin(env, robot, gripper, sponge, p.trajectories)
    # reacher.run(agent_policy=rl_policy)
    env.WaitLoadDone()