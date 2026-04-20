import config
import numpy as np
from scipy.spatial.transform import Rotation as R

def remove_block(block, node, total_layers):
    # Implementation for removing a block from the tower
    x = block.x
    y = block.y
    z = block.z
    ang = block.rotation
    rot_matrix = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    frame_matrix = np.zeros([3,3])
    frame_matrix[0:2, 0:2] = rot_matrix
    frame_matrix[0:2, 2:3] = np.array([[x], [y]])
    frame_matrix[2,2] = 1
    new_frame_matrix = np.linalg.inv(np.array([[0, -1, -0.068], [1, 0, 0.271], [0, 0, 1]])) @ frame_matrix
    x = new_frame_matrix[0, 2]
    y = new_frame_matrix[1, 2]
    rot = R.from_euler('xyz', [180, 0, ang+90], degrees = True).as_quat()
    rot = tuple(rot[[3, 0, 1, 2]])
    print(rot)
    print("x: " + str(x))
    print("y: " + str(y))
    print("z: " + str(0.014*total_layers + 0.1))
    arm_traj = node.plan_arm_to_pose_constraints(
        group_name="arm",
        link_name=config.link_name_real,
        frame_id="world",
        goal_xyz=(0.0, 0.480, 0.1),
        goal_quat_wxyz=rot,
    )
    if arm_traj is not None:
        node.execute_moveit_trajectory(arm_traj)

    print("down")
    arm_traj = node.plan_arm_to_pose_constraints(
        group_name="arm",
        link_name=config.link_name_real,
        frame_id="world",
        goal_xyz=(x, y, z),
        goal_quat_wxyz=rot,
    )
    if arm_traj is not None:
        node.execute_moveit_trajectory(arm_traj)
    
    node.send_gripper_command(
        position=config.gripper_open,
        max_velocity=0.05,
    )
    
    print("up")
    arm_traj = node.plan_arm_to_pose_constraints(
        group_name="arm",
        link_name=config.link_name_real,
        frame_id="world",
        goal_xyz=(x, y, total_layers + 0.1),
        goal_quat_wxyz=rot,
    )
    if arm_traj is not None:
        node.execute_moveit_trajectory(arm_traj)
    
    target_x = config.base_block_locations[block.block_id][1]
    target_y = config.base_block_locations[block.block_id][2]
    
    arm_traj = node.plan_arm_to_pose_constraints(
        group_name="arm",
        link_name=config.link_name_real,
        frame_id="world",
        goal_xyz=(target_x, target_y, total_layers + 0.1),
        goal_quat_wxyz=(0.0, 1.0, 0.0, 0.0),
    )
    if arm_traj is not None:
        node.execute_moveit_trajectory(arm_traj)
    
    arm_traj = node.plan_arm_to_pose_constraints(
        group_name="arm",
        link_name=config.link_name_real,
        frame_id="world",
        goal_xyz=(target_x, target_y, 0.039),
        goal_quat_wxyz=(0.0, 1.0, 0.0, 0.0),
    )
    if arm_traj is not None:
        node.execute_moveit_trajectory(arm_traj)
    
    node.send_gripper_command(
        position=config.gripper_open,
        max_velocity=0.05,
    )
    
    arm_traj = node.plan_arm_to_pose_constraints(
        group_name="arm",
        link_name=config.link_name_real,
        frame_id="world",
        goal_xyz=(target_x, target_y, total_layers + 0.1),
        goal_quat_wxyz=(0.0, 1.0, 0.0, 0.0),
    )
    if arm_traj is not None:
        node.execute_moveit_trajectory(arm_traj)