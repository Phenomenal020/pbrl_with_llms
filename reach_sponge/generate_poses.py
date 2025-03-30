# This class generates starting poses for the robot.
# These are then divided into 3: pre-train poses, train poses, and test poses.
# In total, there are 6 x_pos x 9 z_pos x 4 y_rot = 216 unique poses for initialisation.
# Of these, 16 would be used for pre-training, 180 for training, and 50 for testing.

import numpy as np

class GeneratePoses:
    
    def __init__(self):
        self.pretrain_poses, self.train_poses, self.test_poses = self.generate_poses()

    def generate_poses(self):
        #Define possible values for positions and rotations - Carefully chosen (from  experiments in  the Perception class) to prevent unreasonable initialisations such as initialising the robot on the mannequin or drawer
        np.random.seed(42)
        x_values = np.array([-2.0, -1.75, -1.50, 1.50, 1.75, 2.0])   # 6
        y_pos = 0
        z_values = np.array([-1.50, -1.00, -0.50, 0.0, 0.50, 1.00, 1.50, 2.00, 2.50])  # 9
        x_rot = 0.0
        y_rot_values = np.array([0, 90, 180, 270]) #4. x and z roation values are fixed
        z_rot = 0.0

        # Generate all possible combinations of positions and rotations
        combinations = [
            ([x, y_pos, z], [x_rot, y_rot, z_rot])
            for x in x_values
            for z in z_values
            for y_rot in y_rot_values
        ]  # Expected: 216 combinations

        # First shuffle the array.
        combinations = np.random.permutation(combinations)
        
        # Verify shuffle
        # for i in range(len(combinations)):
        #     print(combinations[i])
        
        # split into three: 16 pre-train poses, 180 train poses, and 50 test poses
        pretrain_poses = combinations[:16]
        train_poses = combinations[16:196]
        # shuffle train_poses and pick 25 to add to test_poses --> 50 test poses
        train_poses_combinations = np.random.permutation(train_poses)
        test_poses = np.concatenate((train_poses_combinations[:25], combinations[191:]), axis=0)
        
        return (pretrain_poses, train_poses, test_poses)    
    
# Test 
if __name__ == "__main__":
    gp = GeneratePoses()
    print(gp.pretrain_poses.shape) 
    print(gp.train_poses.shape)  
    print(gp.test_poses.shape)   