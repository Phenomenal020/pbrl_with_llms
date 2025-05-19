# This class generates starting poses for the robot.
# These are then divided into 3: pre-train poses, train poses, and test poses.
# In total, there are 6 x_pos x 9 z_pos x 4 y_rot = 216 unique poses for initialisation.
# Of these, 16 would be used for pre-training, 180 for training, and 50 available for testing.
# During inference, for comparison sake, make test poses generation deterministic.

import numpy as np

class GeneratePoses:
    
    def __init__(self):
        self.pretrain_poses, self.train_poses, self.test_poses = self.generate_poses()

    def generate_poses(self):
        # Define fixed seed for reproducibility
        seed = 42

        # Define possible values for positions and rotations
        x_values = np.array([-2.0, -1.75, -1.50, 1.50, 1.75, 2.0])  # 6
        y_pos = 0
        z_values = np.array([-1.50, -1.00, -0.50, 0.0, 0.50, 1.00, 1.50, 2.00, 2.50])  # 9
        x_rot = 0.0
        y_rot_values = np.array([0, 90, 180, 270])  # 4
        z_rot = 0.0

        # Generate all possible combinations
        combinations = [
            ([x, y_pos, z], [x_rot, y_rot, z_rot])
            for x in x_values
            for z in z_values
            for y_rot in y_rot_values
        ]  # 6 * 9 * 4 = 216

        # Shuffle all combinations deterministically
        rng = np.random.default_rng(seed)
        combinations = rng.permutation(combinations)

        # Split into subsets
        pretrain_poses = combinations[:16]
        train_poses = combinations[16:196]

        # Shuffle training poses again with same seed (or another fixed one) for test set selection
        rng = np.random.default_rng(seed + 1)  # Use a different but fixed seed to reshuffle train
        train_poses_combinations = rng.permutation(train_poses)

        # Select 25 from shuffled train, and final 25 from original shuffled list
        test_poses = np.concatenate((train_poses_combinations[:25], combinations[191:]), axis=0)

        return (pretrain_poses, train_poses, test_poses)    
    
# Test 
if __name__ == "__main__":
    gp = GeneratePoses()
    print(gp.pretrain_poses.shape)  # (16,)
    print(gp.train_poses.shape)     # (180,)
    print(gp.test_poses.shape)      # (50,)