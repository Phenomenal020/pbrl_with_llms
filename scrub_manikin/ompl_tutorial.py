# source: https://github.com/empriselab/RCareWorld/blob/main/pyrcareworld/pyrcareworld/demo/examples/example_ompl.py

import os  # for file paths
import sys # for system level operations

# This makes sure that Python can locate and import modules from two directories above the current file.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from pyrcareworld.envs.base_env import RCareWorld
import pyrcareworld.attributes as attr
from pyrcareworld.attributes.omplmanager_attr import OmplManagerAttr


try:
    import pyrcareworld.attributes.omplmanager_attr as rfu_ompl
except ImportError:
    raise Exception("This feature requires ompl, see: https://github.com/ompl/ompl")

env = RCareWorld(assets=["franka_panda", "Collider_Box", "OmplManager"])

robot = env.InstanceObject(
    name="franka_panda", id=123456, attr_type=attr.ControllerAttr
)

robot.EnabledNativeIK(False) # native inverse kinematics (IK) is disabled

box1 = env.InstanceObject(name="Collider_Box", id=111111, attr_type=attr.ColliderAttr)
box1.SetTransform(position=[-0.5, 0.5, 0], scale=[0.2, 1, 0.2])
box2 = env.InstanceObject(name="Collider_Box", id=111112, attr_type=attr.ColliderAttr)
box2.SetTransform(position=[0.5, 0.5, 0], scale=[0.2, 1, 0.2])
env.step()

# ompl_manager = env.InstanceObject(name="OmplManager", attr_type=attr.OmplManagerAttr)
ompl_manager = env.InstanceObject(name="OmplManager", attr_type=OmplManagerAttr)
ompl_manager.modify_robot(123456)  # o that it can plan motions for that robot.
env.step()

start_state = [0.0, -45.0, 0.0, -135.0, 0.0, 90.0, 45.0]
# target_state = [ompl_manager.joint_upper_limit[j] * 0.9 for j in range(ompl_manager.joint_num)]
target_state = [
    6.042808,
    -35.73029,
    -128.298,
    -118.3777,
    -40.28789,
    134.8007,
    -139.2552,
]

planner = rfu_ompl.RFUOMPL(ompl_manager, time_unit=5)

print("+++++++++++++++ Working +++++++++++++++")

# begin
ompl_manager.set_state(start_state)
env.step(50)

# target
ompl_manager.set_state(target_state)
env.step(50)

# return
ompl_manager.set_state(start_state)
env.step(50)

# The simulation’s time step is set to a very small value (0.001) and then stepped to register this change.
env.SetTimeStep(0.001)
env.step()

# is_sol is a boolean flag indicating whether a valid solution (path) was found.
# path is the sequence of states (or waypoints) that the planner computed.
is_sol, path = planner.plan_start_goal(start_state, target_state)

# The code prints the target state and the last state in the computed path. This is likely to verify that the planned path ends at the desired target configuration.
print(target_state)
print(path[-1])

# The time step is increased back to 0.02 for executing the path at a normal simulation pace.
env.SetTimeStep(0.02)
env.step()


# if a valid solution was found (is_sol is True), it continuously executes the planned path.
while True:
    if is_sol:
        planner.execute(path)
        
        
# ./install-ompl-ubuntu.sh --github --python