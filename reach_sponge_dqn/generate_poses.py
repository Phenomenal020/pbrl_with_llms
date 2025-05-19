# This class generates starting poses for the robot.
# These are then divided into 2: train poses and test poses.
# In total, there are 2 x_pos x 3 z_pos x 4 y_rot = 24 unique poses for initialisation.
# Of these, 30 will be used for training and 10 for testing.

import numpy as np

class GeneratePoses:
    
    def __init__(self):
        self.train_poses, self.test_poses = self.generate_poses()

    def generate_poses(self):
        
        seed = 42

        # Define possible values for positions and rotations - 40 total
        x_values = np.array([-1.50, 1.50])  # 2
        y_pos = 0
        z_values = np.array([1.00, 1.25, 1.50])  # 3
        x_rot = 0.0
        y_rot_values = np.array([0, 90, 180, 270])  # 4
        z_rot = 0.0

        # Generate all possible combinations
        combinations = [
            ([x, y_pos, z], [x_rot, y_rot, z_rot])
            for x in x_values
            for z in z_values
            for y_rot in y_rot_values
        ]  # 2 * 3 * 4 = 24

        # Shuffle all combinations deterministically
        rng = np.random.default_rng(42)
        combinations = rng.permutation(combinations)

        # Split into subsets
        train_poses = combinations[0:16]
        test_poses = combinations[16:24]

        return (train_poses, test_poses)    
    
# Test 
if __name__ == "__main__":
    gp = GeneratePoses()
    print(gp.train_poses.shape)   
    print("*************************")  
    print(gp.test_poses.shape)      