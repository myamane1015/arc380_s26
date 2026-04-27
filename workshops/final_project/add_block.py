import config
import numpy as np
from scipy.spatial.transform import Rotation as R

def add_block(block, node, numbers, total_layers):
    x = block.x
    y = block.y
    z = block.z
    ang = block.rotation
    x = config.offset_x - block.y + block.layer * 0.006
    y = config.offset_y + block.x - 0.002
    rot = R.from_euler('xyz', [180, 0, ang], degrees = True).as_quat()
    rot = tuple(rot[[3, 0, 1, 2]])
    target_x = config.base_block_locations[numbers[block.block_id]][1]
    target_y = config.base_block_locations[numbers[block.block_id]][2]
    
    arm_traj = node.plan_arm_to_pose_constraints(
        group_name="arm",
        link_name=config.link_name_real,
        frame_id="world",
        goal_xyz=(target_x, target_y, 0.014*(total_layers+1) + 0.032),
        goal_quat_wxyz=(0.0, 1.0, 0.0, 0.0),
    )
    if arm_traj is not None:
        node.execute_moveit_trajectory(arm_traj)
    
    arm_traj = node.plan_arm_to_pose_constraints(
        group_name="arm",
        link_name=config.link_name_real,
        frame_id="world",
        goal_xyz=(target_x, target_y, 0.033),
        goal_quat_wxyz=(0.0, 1.0, 0.0, 0.0),
    )
    if arm_traj is not None:
        node.execute_moveit_trajectory(arm_traj)
    
    node.send_gripper_command(
        position=config.gripper_closed,
        max_velocity=0.05,
    )
    
    arm_traj = node.plan_arm_to_pose_constraints(
        group_name="arm",
        link_name=config.link_name_real,
        frame_id="world",
        goal_xyz=(target_x, target_y, 0.014*(total_layers+1) + 0.032),
        goal_quat_wxyz=(0.0, 1.0, 0.0, 0.0),
    )
    if arm_traj is not None:
        node.execute_moveit_trajectory(arm_traj)
    
    arm_traj = node.plan_arm_to_pose_constraints(
        group_name="arm",
        link_name=config.link_name_real,
        frame_id="world",
        goal_xyz=(x, y, 0.014*(total_layers+1) + 0.032),
        goal_quat_wxyz=rot,
    )
    if arm_traj is not None:
        node.execute_moveit_trajectory(arm_traj)
    
    arm_traj = node.plan_arm_to_pose_constraints(
        group_name="arm",
        link_name=config.link_name_real,
        frame_id="world",
        goal_xyz=(x, y, z+0.003),
        goal_quat_wxyz=rot,
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
        goal_xyz=(x, y, 0.014*(total_layers+1) + 0.032),
        goal_quat_wxyz=rot,
    )
    if arm_traj is not None:
        node.execute_moveit_trajectory(arm_traj)