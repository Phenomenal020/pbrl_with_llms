import json

class GenTrajectory:
    
    def __init__(self):
        # Read in the landmarks from the landmarks.json file
        try:
            with open('landmarks.json', 'r') as f:
                self.landmarks = json.load(f)
        except Exception as e:
            print("Error reading landmarks.json:", e)
            self.landmarks = {}
        
        # Placeholder to correct for depth mismatch. If not None, it will be added to each z coordinate.
        self.depth_error = None
        
    def generate_trajectory(self):
        # Generate a list of trajectories to cover:
        # right_arm: [right_shoulder -> right_index -> right_shoulder]
        # left_arm: [left_shoulder -> left_index -> left_shoulder]
        # right_leg: [right_hip -> right_foot_index -> right_hip]
        # left_leg: [left_hip -> left_foot_index -> left_hip]
        # central_body: [right_shoulder -> left_shoulder -> left_hip -> right_hip -> right_shoulder]
        
        trajectories = {}
        
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
            print(f"Missing landmark in the data: {e}")
            return trajectories
        
        # Build each trajectory as a list of waypoints
        trajectories["right_arm"] = [right_shoulder, right_index, right_shoulder]
        trajectories["left_arm"] = [left_shoulder, left_index, left_shoulder]
        trajectories["right_leg"] = [right_hip, right_foot_index, right_hip]
        trajectories["left_leg"] = [left_hip, left_foot_index, left_hip]
        trajectories["central_body"] = [right_shoulder, left_shoulder, left_hip, right_hip, right_shoulder]
        
        # If depth_error correction is provided, adjust the z-axis of each waypoint.
        if self.depth_error is not None:
            for traj_key, traj in trajectories.items():
                for i, point in enumerate(traj):
                    # Assuming the depth (z-axis) is the third coordinate
                    corrected_point = point.copy()
                    corrected_point[2] += self.depth_error
                    traj[i] = corrected_point
        
        # The trajectories are formatted for use in an OMPL planner,
        # where each trajectory is a list of [x, y, z] states.
        return trajectories

# Example usage
if __name__ == "__main__":
    traj_generator = GenTrajectory()
    trajectories = traj_generator.generate_trajectory()
    print("Generated Trajectories:")
    for key, traj in trajectories.items():
        print(f"{key}: {traj}")
