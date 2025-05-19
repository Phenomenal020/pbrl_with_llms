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
    def get_dims(self):
        # Get a dictionary to save the 4 corners of the objects
        bboxes = {}
        
        self.env.step(50)
        
        # Capture an ID map image
        self.scene_camera.GetID(512, 512)
        self.env.step()
        id_map = np.frombuffer(self.scene_camera.data["id_map"], dtype=np.uint8)
        id_map = cv2.imdecode(id_map, cv2.IMREAD_COLOR)
        
        # Get the 2d bbox data from the image
        self.scene_camera.Get2DBBox(512, 512)
        self.env.step()
              
        self.scene_camera.Get3DBBox()
        self.env.step()
        for i in self.scene_camera.data['3d_bounding_box']:
                print(i)
                print(self.scene_camera.data['3d_bounding_box'][i])
                
        
        
if __name__ == "__main__":

    # Initialise the environment
    env = BathingEnv(graphics=True)  
    
    # Create a perception object
    p = Perception(env)
    
    env.AlignCamera(123456)  # Scene camera
    
    robot = p.get_robot()
    sponge = p.get_sponge()
    gripper = p.get_gripper()
    
    gripper_pos = [robot.data["position"][0], sponge.data.get("position")[1] + 0.3, robot.data["position"][2]]
    robot.IKTargetDoMove(
            position=gripper_pos,
            duration=1,
            speed_based=False,
        )
    robot.WaitDo()
    env.step(50)

    robot.SetPosition([-1.5, 0, 1.00])
    robot.SetRotation([0, 0, 0])
    env.step(10)
    
    # robot.MoveForward(1.0, 0.5)
    # for i in range(100):
    #     print(robot.data.get("position"))
    #     env.step()
    
    p.get_dims()
    
    # Do not close window
    env.WaitLoadDone()