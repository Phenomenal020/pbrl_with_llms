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
    
    positions = [
        [-1.7520437240600586, 0.009195610880851746, -2.1457672119140625e-06],
        [1.501564383506775, 0.009212583303451538, -1.5000007152557373],
        [-2.0019383430480957, 0.009212106466293335, 2.500002861022949],
        [-1.7499996423721313, 0.009175516664981842, 1.4980335235595703],
        [-1.5000003576278687, 0.009226799011230469, 2.5016028881073],
        [2.0000007152557373, 0.009227782487869263, 1.4980605840682983],
        [1.999999761581421, 0.009152933955192566, -0.998075008392334],
        [-1.5016554594039917, 0.009233377873897552, 2.5000009536743164],
        [-1.9980664253234863, 0.009249009191989899, -1.000004529953003],
        [-1.7499994039535522, 0.009121187031269073, 1.998131513595581],
        [1.5000007152557373, 0.0092385932803154, -0.9982655048370361],
        [-1.499999761581421, 0.009276479482650757, 2.0019211769104004],
        [-1.5000020265579224, 0.00909966230392456, 2.001795530319214],
        [1.5000005960464478, 0.009238183498382568, -1.4981858730316162],
        [-2.0000011920928955, 0.009309515357017517, 0.49813544750213623],
        [-1.9999995231628418, 0.009071782231330872, 0.9982818365097046]
    ]
    
    rotations = [
        [359.8263244628906, 269.96795654296875, 0.06436479836702347],
        [359.8438415527344, 90.01119995117188, 0.1627093255519867],
        [359.8306579589844, 270.00677490234375, 359.7912902832031],
        [359.828857421875, 179.9658203125, 0.0396994911134243],
        [359.8423156738281, 0.008096595294773579, 0.17540289461612701],
        [359.8307189941406, 180.0056915283203, 359.8012390136719],
        [359.83013916015625, 359.96417236328125, 0.015876106917858124],
        [359.8406677246094, 270.0038146972656, 0.18819881975650787],
        [359.83123779296875, 90.005126953125, 359.81353759765625],
        [359.83203125, 179.9657440185547, 359.99322509765625],
        [359.8380432128906, 0.0015233138110488653, 0.19801300764083862],
        [359.83135986328125, 0.010920396074652672, 359.8296203613281],
        [359.83441162109375, 359.964111328125, 359.9680480957031],
        [359.83526611328125, -0.00148802122566849, 0.20542383193969727],
        [359.83367919921875, 180.01101684570312, 359.8481750488281],
        [359.8370056152344, 179.96531677246094, 359.9422912597656]
    ]
    
    for pos, rot in zip(positions, rotations):
        print(f"Testing Position: {pos} and Rotation: {rot}")
        robot.SetPosition(pos)
        robot.SetRotation(rot)
        env.step(50)
        
        # Raise the gripper to a safe height for easy obstacle avoidance
        gripper_pos = [robot.data["position"][0], sponge.data.get("position")[1] + 0.3, robot.data["position"][2]]
        robot.IKTargetDoMove(
            position=gripper_pos,
            duration=1,
            speed_based=False,
        )
        robot.WaitDo()
        env.step(50)
        gripper.GripperOpen()
        env.step(50)
        
        # Lower gripper (Grasp sponge)
        lower_position = [sponge.data.get("position")[0], sponge.data.get("position")[1]+0.03, sponge.data.get("position")[2]]
        robot.IKTargetDoMove(
            position=lower_position,  # Move directly above the sponge to grasp it
            duration=1,
            speed_based=False,
        )
        robot.WaitDo()
        env.step(50)
    
    env.WaitLoadDone()