import json

class GenTrajectory:
    
    def __init__(self):
        # Read in the landmarks from the landmarks.json file
        try:
            with open('landmarks.json', 'r') as f:
                self.landmarks = json.load(f)
        except Exception as e:
            raise Exception("Error reading landmarks.json:", e)
        
        # Placeholder to correct for depth mismatch. If not None, it will be added to each z coordinate.
        self.depth_error = None
        
    def generate_trajectory(self):
        trajectories = []
        
        try:
            # Extract the necessary landmarks
            right_shoulder = self.landmarks["right_shoulder"]
            right_index    = self.landmarks["right_index"]
            left_shoulder  = self.landmarks["left_shoulder"]
            left_index     = self.landmarks["left_index"]
            right_hip      = self.landmarks["right_hip"]
            right_foot_index = self.landmarks["right_foot_index"]
            left_hip       = self.landmarks["left_hip"]
            left_foot_index  = self.landmarks["left_foot_index"]
        except KeyError as e:
            raise Exception(f"Missing landmark in the data: {e}")
        
        trajectories.append(right_index)
        trajectories.append(right_shoulder)
        trajectories.append(right_hip)
        trajectories.append(right_foot_index)
        trajectories.append(left_index)
        trajectories.append(left_shoulder)
        trajectories.append(left_hip)
        trajectories.append(left_foot_index)
        
        # TODO: Place holder for depth error correction
        if self.depth_error is not None:
            return trajectories


# # Example usage
# if __name__ == "__main__":
#     traj_generator = GenTrajectory()
#     trajectories = traj_generator.generate_trajectory()
#     print("Generated Trajectories:")
#     for key, traj in trajectories.items():
#         print(f"{key}: {traj}")
