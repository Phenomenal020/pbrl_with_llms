# Perception class:
# To get the locations of objects in the scene.
# This file was mostly used during the initial stages to know the locations of objects. Subsequently, the code works fine without it.

from pyrcareworld.envs.bathing_env import BathingEnv
import pyrcareworld.attributes.camera_attr as attr
import numpy as np
import cv2

import mediapipe as mp 
import json


class Perception:
    
    # Setup a camera to capture objects in the scene
    def __init__(self, env:BathingEnv):
        # Get an instance of the environment
        self.env = env  
        
        # Initialise the holistic model and drawing tools (for confirmation)
        self.holistic = mp.solutions.holistic.Holistic(
            static_image_mode=True,
            model_complexity=2,
            refine_face_landmarks=True,
            enable_segmentation=False
        )
        self.drawing_utils = mp.solutions.drawing_utils
        self.drawing_styles = mp.solutions.drawing_styles
        
        # Setup a camera to capture the entire scene
        self.scene_camera = env.InstanceObject(name="Camera", id=123456, attr_type=attr.CameraAttr) 
        # Set position and orientation for the scene camera
        self.scene_camera.SetTransform(position=[0, 4.2, 0.6], rotation=[90, 0, 0.0])   
        
        # Setup a camera to get manikin data
        self.manikin_camera = env.InstanceObject(name="Camera", id=333333, attr_type=attr.CameraAttr)
        self.manikin_camera.SetTransform(position=[-0.10, 2.0, 0.5], rotation=[90, 0, 0])
        # self.manikin_camera.SetTransform(position=[0.105, 1.7, 1.7], rotation=[90, 0, 0])
        # Capture depth data from the manikin camera
        self.env.step(1)

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
    
    # Get Water tank and SPonge Platform (id = 758666) from camera for obstacle avoidance
    def get_platform(self):
        coords = self._get_coords(758666)
        return coords
    
    # Get bed (id = 758554) from camera for obstacle avoidance. Used to determine bed's location for robot initialisation. This is to ensure the robot doesn't get initialised IN THE BED
    def get_bed(self):
        coords = self._get_coords(758554)
        return coords
    
    
    # 3D bounding box information. Used to determine bed's location for robot initialisation
    def _get_coords(self, id):
        # Capture and save ID map image
        self.scene_camera.GetID(512, 512)
        env.step()
        id_map = np.frombuffer(p.scene_camera.data["id_map"], dtype=np.uint8)
        id_map = cv2.imdecode(id_map, cv2.IMREAD_COLOR)
        cv2.imwrite("id_map.png", id_map)
        
        # Use the id param to get the pixel coordinates of the relevant object from the image
        self.scene_camera.Get3DBBox()
        env.step()
        coords = None
        for i in p.scene_camera.data["3d_bounding_box"]:
            if i == id:
                coords = self.scene_camera.data["3d_bounding_box"][i]
        return coords
    
    
    # Used to get location information of objects. Used to determine bed's location for robot initialisation
    def _get_2dbbox(self):
        # Get a dictionary to save the 4 corners of the objects
        bboxes = {}
        
        # Capture an ID map image
        self.scene_camera.GetID(512, 512)
        self.env.step()
        id_map = np.frombuffer(self.scene_camera.data["id_map"], dtype=np.uint8)
        id_map = cv2.imdecode(id_map, cv2.IMREAD_COLOR)
        
        # Get Bthe 2d bbox data from the image
        self.scene_camera.Get2DBBox(512, 512)
        self.env.step()
        
        for obj_id, bbox_data in self.scene_camera.data["2d_bounding_box"].items():            
        #     # Extract data
            center = bbox_data[0:2]
            size = bbox_data[2:4]
            
        #     # Calculate top-left (tl) and bottom-right (br) points
            tl_point = (int(center[0] - size[0] / 2), int(512 - center[1] - size[1] / 2))
            br_point = (int(center[0] + size[0] / 2), int(512 - center[1] + size[1] / 2))
            
        #     # Save the bbox information
            bboxes[obj_id] = {
                "top_left": tl_point,
                # "width": int(size[0]),
                # "height": int(size[1]),
                "bottom_right": br_point,
            }
            
        print(f"bboxes: {bboxes}")
        
     
    # Source: https://medium.com/@speaktoharisudhan/media-pipe-exploring-holistic-model-32b851901f8a   
    def get_poses(self):
        
        # Capture RGB image from manikin camera
        manikin = self.manikin_camera.GetRGB(512, 512)
        self.env.step()
        manikin_image = np.frombuffer(manikin.data["rgb"], dtype=np.uint8).reshape(512, 512, 3)
        manikin_image = cv2.cvtColor(manikin_image, cv2.COLOR_BGR2RGB)

        # Process the image using the Holistic model.
        results = self.holistic.process(manikin_image)

        # Store the landmarks in a dictionary 
        landmarks_dict = {}

        # Extract body landmarks
        if results.pose_landmarks:
             landmarks_dict['pose'] = [
                {
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                }
                for landmark in results.pose_landmarks.landmark
            ]

        # # Extract face landmarks
        if results.face_landmarks:
             landmarks_dict['face'] = [
                {
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z
                }
                for landmark in results.face_landmarks.landmark
            ]

        # Extract left hand landmarks
        if results.left_hand_landmarks:
             landmarks_dict['left_hand'] = [
                {
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z
                }
                for landmark in results.left_hand_landmarks.landmark
            ]

        # Extract right hand landmarks
        if results.right_hand_landmarks:
             landmarks_dict['right_hand'] = [
                {
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z
                }
                for landmark in results.right_hand_landmarks.landmark
            ]
            
        # Draw pose, face, and hands landmarks on the frame and save as 3d_pose.png
        # Convert RGB image back to BGR for OpenCV visualization
        image_bgr = cv2.cvtColor(manikin_image, cv2.COLOR_RGB2BGR)

        # Draw pose, face, and hand landmarks
        if results.pose_landmarks:
            self.drawing_utils.draw_landmarks(
                image_bgr,
                results.pose_landmarks,
                mp.solutions.holistic.POSE_CONNECTIONS,
                self.drawing_styles.get_default_pose_landmarks_style()
            )

        if results.face_landmarks:
            self.drawing_utils.draw_landmarks(
                image_bgr,
                results.face_landmarks,
                mp.solutions.holistic.FACEMESH_TESSELATION,
                self.drawing_styles.get_default_face_mesh_tesselation_style()
            )

        if results.left_hand_landmarks:
            self.drawing_utils.draw_landmarks(
                image_bgr,
                results.left_hand_landmarks,
                mp.solutions.holistic.HAND_CONNECTIONS,
                self.drawing_styles.get_default_hand_landmarks_style()
            )

        if results.right_hand_landmarks:
            self.drawing_utils.draw_landmarks(
                image_bgr,
                results.right_hand_landmarks,
                mp.solutions.holistic.HAND_CONNECTIONS,
                self.drawing_styles.get_default_hand_landmarks_style()
            )

        # Save annotated image
        cv2.imwrite("3d_pose.png", image_bgr)

        # Print the output in JSON format so it can be used by a robot or another processing system.
        print(json.dumps( landmarks_dict, indent=2))


if __name__ == "__main__":

    # Initialise the environment
    env = BathingEnv(graphics=True)  
    
    # Create a perception object
    p = Perception(env)
    
    env.AlignCamera(123456)  # Scene camera
    # env.AlignCamera(333333) # Mannequin camera
    
    robot = p.get_robot()
    sponge = p.get_sponge()
    gripper = p.get_gripper()

    robot.SetPosition([-0.1, 0, 1.65])
    robot.SetRotation([357, 269, 0])

    # Raise the gripper to a safe height for easy obstacle avoidance
    gripper_pos = [robot.data["position"][0], sponge.data.get("position")[1] + 0.3, robot.data["position"][2]]
    robot.IKTargetDoMove(
        position=gripper_pos,
        duration=0,
        speed_based=False,
    )
    robot.WaitDo()
    env.step(50)
    gripper.GripperOpen()
    env.step(50)
    
    # robot.MoveForward(0.27, 2)
    # env.step(100)
    
    # robot.TurnLeft(90, 1)
    # env.step(300)
    
    # robot.MoveForward(0.2, 2)
    # env.step(50)
    
    print(robot.data.get("position"))
    print(robot.data.get("position"))
    
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
    
    robot.MoveForward(0.2, 2)
    env.step(50)
    
    # p.get_poses()
    
    
    # TESTBED FOR DIP SPONGE (TBC)-------------------------------
    # Raise the gripper to a safe height for easy obstacle avoidance
    # gripper_pos = [robot.data["position"][0], sponge.data.get("position")[1] + 0.5, robot.data["position"][2]]
    # robot.IKTargetDoMove(
    #     position=gripper_pos,
    #     duration=0,
    #     speed_based=False,
    # )
         
    # # Open the gripper and set its orientation to be the same with that of the robot
    # gripper.SetRotation(robot.data.get("rotation"))
    # gripper.GripperOpen()
    # robot.IKTargetDoComplete()
    # robot.WaitDo()
    # env.step()
    
    # robot.SetPosition([-0.1, 0.0, 1.72])
    # robot.SetRotation([357, 266, 359])
    
    # initial_sponge_pos = sponge.data.get("position")
    
    # # Lower gripper (Grasp sponge)
    # lower_position = [sponge.data.get("position")[0], sponge.data.get("position")[1]+0.03, sponge.data.get("position")[2]]
    # robot.IKTargetDoMove(
    #     position=lower_position,  # Move directly above the sponge to grasp it
    #     duration=1,
    #     speed_based=False,
    # )
    # robot.WaitDo()
    # env.step(50)
    
    # initial_gripper_pos = gripper.data.get("position")
    
    # gripper.GripperClose()
    # env.step(50)
    
    # # Raise gripper
    # gripper_pos = [initial_gripper_pos[0], initial_sponge_pos[1] + 0.4, initial_gripper_pos[2]]
    # robot.IKTargetDoMove(
    #     position=gripper_pos,
    #     duration=1,
    #     speed_based=False,
    # )
    # robot.WaitDo()
    # env.step(50)
    
    #  # Move back
    # robot.MoveBack(0.15, 1)  # Predefined distance between sponge and center of water tank
    # env.step(50)
    # robot.MoveBack(0.15, 1)  # Predefined distance between sponge and center of water tank
    # env.step(50)
    
    # # Lower gripper (dip sponge)
    # lower_position = [initial_gripper_pos[0], initial_sponge_pos[1]+0.03, initial_gripper_pos[2]]
    # robot.IKTargetDoMove(
    #     position=lower_position,  # Move directly above the sponge to grasp it
    #     duration=1,
    #     speed_based=False,
    # )
    # robot.WaitDo()
    # env.step(50)
    
    # # Raise gripper
    # gripper_pos = [initial_gripper_pos[0], sponge.data.get("position")[1] + 0.5, initial_gripper_pos[2]]
    # robot.IKTargetDoMove(
    #     position=gripper_pos,
    #     duration=1,
    #     speed_based=False,
    # )
    # robot.WaitDo()
    # env.step(50)
                
    
    # robot.TurnRight(90, 1)
    # env.step(300)
    
    # # Move back
    # robot.MoveBack(0.15, 1)  # Predefined distance between sponge and center of water tank
    # env.step(50)
    # robot.MoveBack(0.15, 1)  # Predefined distance between sponge and center of water tank
    # env.step(50)
    
    # # Move back
    # robot.MoveBack(0.15, 1)  # Predefined distance between sponge and center of water tank
    # env.step(50)
    # robot.MoveBack(0.15, 1)  # Predefined distance between sponge and center of water tank
    # env.step(50)
    
    # robot.MoveBack(0.15, 1)  # Predefined distance between sponge and center of water tank
    # env.step(50)
    
    # robot.TurnRight(90, 1)
    # env.step(300)
    
    
    

    # print(robot.data.get("position"))
    # print(robot.data.get("rotation"))
    # print(robot.data.get("rotation"))
    
    # # Capture and save RGB image
    # p.manikin_camera.GetRGB(2048, 2048)
    # env.step()
    # rgb = np.frombuffer(p.manikin_camera.data["rgb"], dtype=np.uint8)
    # rgb = cv2.imdecode(rgb, cv2.IMREAD_COLOR)
    # cv2.imwrite("reach_manikin.png", rgb)
    
    # Do not close window
    env.WaitLoadDone()