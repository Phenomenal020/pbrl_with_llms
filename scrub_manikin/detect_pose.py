# source: Tutorial & code
# https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/pose_landmarker/python/%5BMediaPipe_Python_Tasks%5D_Pose_Landmarker.ipynb#scrollTo=s3E6NFV-00Qt

# https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python

# The output contains the following world coordinates (WorldLandmarks):
#     x, y, and z: Real-world 3-dimensional coordinates in meters, with the midpoint of the hips as the origin.
#     visibility: The likelihood of the landmark being visible within the image.


# Import necessary libraries
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image, ImageFormat

import numpy as np


class DetectPose:
  
  def __init__(self, image):
    # Create a PoseLandmarker object.
    base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False)  # Disable segmentation mask output
    self.image = image
    self.detector = vision.PoseLandmarker.create_from_options(options)
    
  
  def get_landmarks(self):
    # Load the input image from a numpy array.
    mp_image = Image(image_format=ImageFormat.SRGB, data=self.image)
    
    # detect poses
    detection_result = self.detector.detect(mp_image)
    
    # print(f"Detection result: {detection_result}")

    # # Process the detection result. In this case, visualise it.
    # annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)
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
    right_index = world_landmarks[20]
    right_shoulder = world_landmarks[12]
    # left arm
    left_shoulder = world_landmarks[11]
    left_index = world_landmarks[19]
    # right leg
    right_hip = world_landmarks[24]
    right_foot_index = world_landmarks[32]
    # left leg
    left_hip = world_landmarks[23]
    left_foot_index = world_landmarks[31]
    # mid-body
    # mid_body = (right_shoulder, left_shoulder, left_hip, right_hip, right_shoulder)
    
    return True, {
      "right_index": right_index,
      "right_shoulder": right_shoulder,
      "left_shoulder": left_shoulder,
      "left_index": left_index,
      "right_hip": right_hip,
      "right_foot_index": right_foot_index,
      "left_hip": left_hip,
      "left_foot_index": left_foot_index,
    }

# if __name__ == "__main__":
#   # resize image 
#   img = cv2.imread("assets/manikin.png")
#   image = cv2.resize(img, None, fx=0.5, fy=0.5)
  
#   # Create a DetectPose instance
#   pd = DetectPose()
    
#   # Detect pose landmarks from the input image.
#   detection_result = pd.get_landmarks(image)
  
#   # verify
#   # print(detection_result)