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

# import pyrcareworld.attributes as _attr
from pyrcareworld.attributes.omplmanager_attr import OmplManagerAttr

try:
    import pyrcareworld.attributes.omplmanager_attr as rfu_ompl
except ImportError:
    raise Exception("This feature requires ompl, see: https://github.com/ompl/ompl")


class Perception:
    
    # Setup a camera to capture objects in the scene
    def __init__(self, env:BathingEnv):
        # Get an instance of the environment
        self.ompl_manager = env.InstanceObject(name="OmplManager", attr_type=OmplManagerAttr)
        self.env = env  
        
        # # Initialise the holistic model and drawing tools (for confirmation)
        # self.holistic = mp.solutions.holistic.Holistic(
        #     static_image_mode=True,
        #     model_complexity=2,
        #     refine_face_landmarks=True,
        #     enable_segmentation=False
        # )
        # self.drawing_utils = mp.solutions.drawing_utils
        # self.drawing_styles = mp.solutions.drawing_styles
        
        # Setup a camera to capture the entire scene
        self.scene_camera = env.InstanceObject(name="Camera", id=123456, attr_type=attr.CameraAttr) 
        # Set position and orientation for the scene camera
        self.scene_camera.SetTransform(position=[0, 4.2, 0.6], rotation=[90, 0, 0.0])   
        
        # Setup a camera to get manikin data
        self.manikin_camera = env.InstanceObject(name="Camera", id=333333, attr_type=attr.CameraAttr)
        self.manikin_camera.SetTransform(position=[0.0, 2.1, 0.15], rotation=[88, 0, 0])
        
        self.test_camera = env.InstanceObject(name="Camera", id=444444, attr_type=attr.CameraAttr)
        
    
        
        # # right shoulder - ✅
        # self.test_camera.SetTransform(position=[
        #     0.4879406988620758,
        #     2.165602606022067,
        #     0.018232385705884813
        # ], rotation=[88, 0, 0])
        
        # right index - ✅
        # self.test_camera.SetTransform(position=[
        #     -0.020131396129727364,
        #     2.3013671402469313,
        #     -0.2289543200973192
        # ], rotation=[88, 0, 0])
        
        # left shoulder - ✅
        self.test_camera.SetTransform(position=[
            0.4253469407558441,
            2.1967452846284026,
            0.31049588054022736
        ], rotation=[88, 0, 0])
        
        # left index - ✅
        # self.test_camera.SetTransform(position=[
        #     -0.060410916805267334,
        #     2.166968099542488,
        #     0.4737346650066623
        # ],rotation=[88, 0, 0])
        
          # right hip - ✅
        # self.test_camera.SetTransform(position=[
        #     0.009450765326619148,
        #     2.1277850183238614,
        #     0.05627584488709447
        # ],  rotation=[88, 0, 0])
        
        # left hip - ✅
        # self.test_camera.SetTransform(position=[
        #     -0.007030941080302,
        #     2.070760824181983,
        #     0.24616934685008607
        # ], rotation=[88, 0, 0])
        
        # right foot index - ✅
        # self.test_camera.SetTransform(position=[
        #     -0.7526283860206604,
        #     1.9350535908759516,
        #     0.05858582278477886
        # ], rotation=[88, 0, 0])
        
        # left  foot index - ✅
        # self.test_camera.SetTransform(position=[
        #     -0.8924819231033325,
        #     1.9566713603890182,
        #     0.26285394538570106
        # ], rotation=[88, 0, 0])
        
        self.landmarks_dict = {}
        self.l2w_poses= {}
        self.env.step(50)

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
            self.l2w_poses[key] = world_point[:3]




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

    # # --------------- Grasp sponge and Dip water tank --------------------
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
        position=lower_position,  # Move directly above the sponge to grasp it
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
    
    # Lower Gripper to dip sponge
    robot.IKTargetDoMove(
        position=lower_position,  # Move directly above the sponge to grasp it
        duration=1,
        speed_based=False,
    )
    robot.WaitDo()
    env.step(50)
    
     # Raise Gripper
    robot.IKTargetDoMove(
        position=gripper_pos,
        duration=1,
        speed_based=False,
    )
    robot.WaitDo()
    env.step(50)
    
    # Turn to face Manikin
    robot.TurnRight(270, 1.5)
    env.step(500)
    
    # env.AlignCamera(333333) # Mannequin camera
    # Switch to test camera
    env.AlignCamera(333333) # Mannequin camera
    
    # Move towards Manikin
    robot.MoveForward(1, 2)
    env.step(150)
    
    robot.TurnLeft(90, 1.5)
    env.step(225)
    
    # # --------------------------------------------------------------- Scrub manikin perception task --------------------------------------------------------------------
    # # Capture an image of the manikin from the manikin camera
    # image = p.manikin_camera.GetRGB(width=512, height=512)
    # env.step()
    # rgb = np.frombuffer(p.manikin_camera.data["rgb"], dtype=np.uint8)
    # rgb = cv2.imdecode(rgb, cv2.IMREAD_COLOR)
    
    # # image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    # # cv2.imshow("Manikin Camera View", image)
    # # cv2.waitKey(0)
    # # cv2.destroyAllWindows()
    
    # # # Detect pose
    # pd = DetectPose()
    # lmarks = pd.get_landmarks(rgb)
    # # print(lmarks)
    
    # # convert lmarks to homogeneous coordinates
    # p.convert_to_homcoords(lmarks)
    
    # # Get the camera's local to world matrix
    # local_to_world_matrix = p.manikin_camera.data.get("local_to_world_matrix")
    
    
    # # convert to world coordinates
    # p.convert_to_world_coords(local_to_world_matrix)
    
    # # Save the l2w poses to a json file for further investigation
    # converted_poses = {key: value.tolist() for key, value in p.l2w_poses.items()}
    # with open("landmarks.json", "w") as file:
    #     json.dump(converted_poses, file, indent=4)
    
    # # # Testing world coordinates transformation
    # env.AlignCamera(444444)
    
    
    
    
    
    # -------------------------------------- OMPL Integration -----------------------------
    robot.EnabledNativeIK(False) # native inverse kinematics (IK) is disabled
    env.step()
    
    # p.ompl_manager = env.InstanceObject(name="OmplManager", attr_type=_attr.OmplManagerAttr)
    p.ompl_manager.modify_robot(221582)  # o that it can plan motions for that robot.
    env.step()
    
    planner = rfu_ompl.RFUOMPL(p.ompl_manager, time_unit=5)

    print("+++++++++++++++ OMPL Working +++++++++++++++")
    
    start_state_cs = [
        -0.020131396129727364,
        2.3013671402469313,
        -0.2289543200973192
    ]
    robot.GetIKTargetJointPosition(start_state_cs, iterate=100)
    env.step()
    start_state_js =  robot.data['result_joint_position']
    print(f"++++++ {start_state_js} +++++++ ")
    
    target_state_cs = [
        0.4879406988620758,
        2.165602606022067,
        0.018232385705884813
    ]
    robot.GetIKTargetJointPosition(target_state_cs, iterate=100)
    env.step()
    target_state_js =  robot.data["result_joint_position"]
    print(f"++++++ {target_state_js} +++++++ ")

    # begin
    p.ompl_manager.set_state(start_state_js)
    env.step(50)

    # target
    p.ompl_manager.set_state(target_state_js)
    env.step(50)

    # return
    p.ompl_manager.set_state(start_state_js)
    env.step(50)

    # The simulation’s time step is set to a very small value (0.001) and then stepped to register this change.
    env.SetTimeStep(0.001)
    env.step()

    # is_sol is a boolean flag indicating whether a valid solution (path) was found.
    # path is the sequence of states (or waypoints) that the planner computed.
    is_sol, path = planner.plan_start_goal(target_state_js, target_state_js)

    # The code prints the target state and the last state in the computed path. This is likely to verify that the planned path ends at the desired target configuration.
    print(target_state_js)
    print(path[-1])

    # The time step is increased back to 0.02 for executing the path at a normal simulation pace.
    env.SetTimeStep(0.02)
    env.step()


    # if a valid solution was found (is_sol is True), it continuously executes the planned path.
    while True:
        if is_sol:
            planner.execute(path)
    
    
    
    # Do not close window
    env.WaitLoadDone()
    
    # self.data[‘result_joint_position’]