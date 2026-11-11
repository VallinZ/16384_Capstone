from xarm7_lib import Robot
# from libcapstone import LightsaberDetector
import numpy as np
from scipy.spatial.transform import Rotation

np.set_printoptions(precision=2, suppress=True)

link_length = [0, 0, 0, 52.5, 77.5, 0, 76]
link_twist = [0, -np.pi/2, np.pi/2, np.pi/2, np.pi/2, np.pi/2, -np.pi/2]
link_offset = [267, 0, 293, 0, 342.5, 0, 269]
lower_limits = np.array([-2*np.pi, -2.059, -2*np.pi, -0.1919, -2*np.pi, -1.6929, -2*np.pi])
upper_limits = np.array([2*np.pi,  2.0944,  2*np.pi,  3.9270,  2*np.pi,  3.1415,  2*np.pi])

def transX(x):
    return np.array([[1, 0, 0, x],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])

def transZ(z):
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, z],
                     [0, 0, 0, 1]])

def rotX(theta):
    return np.array([[1, 0, 0, 0],
                     [0, np.cos(theta), -np.sin(theta), 0],
                     [0, np.sin(theta), np.cos(theta), 0],
                     [0, 0, 0, 1]])

def rotZ(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0, 0],
                     [np.sin(theta), np.cos(theta), 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])

def forward_kinematics(joint_angles):
    # in mm
    dof = 7
    frames = np.zeros((4, 4, dof + 1))
    current_transform = np.eye(4)
    frames[:, :, 0] = current_transform
    for i in range(7):
        Hi = rotX(link_twist[i]) @ transX(link_length[i]) @ rotZ(joint_angles[i]) @ transZ(link_offset[i])
        current_transform =  current_transform @ Hi
        frames[:, :, i + 1] = current_transform
    return frames

def current_pose(thetas):
    EE_transform = forward_kinematics(thetas)[:, :, -1]
    current_pos = EE_transform[:3, 3]
    R = EE_transform[:3, :3]
    current_rotvec = Rotation.from_matrix(R).as_rotvec()
    current_pose = np.concatenate([current_pos, current_rotvec])
    return current_pose

def ee(thetas):
    EE_transform = forward_kinematics(thetas)[:, :, -1]
    return EE_transform

def jacobians(thetas):
    """
    Compute the Jacobians for each frame.

    Parameters:
    thetas (np.ndarray): Joint angles, shape (dof,)

    Returns:
    np.ndarray: Jacobians, shape (4, dof, dof+1)
    """
    dof = 7
    thetas = np.asarray(thetas)
    if thetas.shape != (dof,):
        raise ValueError(f'Invalid thetas: Expected shape ({dof},), got {thetas.shape}.')

    jacobians = np.zeros((6, dof, dof + 1))
    epsilon = 0.001
    frames = forward_kinematics(thetas)

    for j in range(dof):
        thetas_q = thetas.copy()
        thetas_q[j] += epsilon
        frames_q = forward_kinematics(thetas_q)
        for i in range(dof + 1):
            dx = (frames_q[0, 3, i] - frames[0, 3, i]) / epsilon
            dy = (frames_q[1, 3, i] - frames[1, 3, i]) / epsilon
            dz = (frames_q[2, 3, i] - frames[2, 3, i]) / epsilon
            jacobians[0, j, i] = dx
            jacobians[1, j, i] = dy
            jacobians[2, j, i] = dz
            R = frames[:3, :3, i]
            Rq = frames_q[:3, :3, i]
            
            R_rel = Rq @ R.T
            
            rotvec = Rotation.from_matrix(R_rel).as_rotvec()
            omega = rotvec / epsilon
            
            jacobians[3, j, i] = omega[0]
            jacobians[4, j, i] = omega[1]
            jacobians[5, j, i] = omega[2]
    return jacobians

def inverse_kinematics(initial_thetas, goal_position_or_pose):
    """
    Returns the joint angles which achieve either a 3D position goal [x, y, z]
    or a 6D pose goal [x, y, z, wx, wy, wz] using a spatial Jacobian.

    Parameters:
    initial_thetas (np.ndarray): Initial joint angles, shape (dof,)
    goal_position_or_pose (np.ndarray): Goal vector, shape (3,) or shape (6,)

    Returns:
    np.ndarray: Joint angles that achieve the goal.
    """
    dof = 7 
    initial_thetas = np.asarray(initial_thetas)
    goal = np.asarray(goal_position_or_pose)
    
    if initial_thetas.shape != (dof,):
        raise ValueError(f'Invalid initial_thetas: Expected shape ({dof},), got {initial_thetas.shape}.')

    if goal.shape[0] not in [3, 6]:
        raise ValueError(f'Invalid goal shape: Expected length 3 or 6, got {goal.shape}.')

    thetas = initial_thetas.copy()
    step_size = 0.1       
    stopping_condition = 1e-4 
    max_iter = 5000
    
    for num_iter in range(max_iter):
        EE_transform = ee(thetas)
        current_pos = EE_transform[:3, 3]
        J = jacobians(thetas)[:, :, -1]  
        current_pos = EE_transform[:3, 3]
        R_curr = EE_transform[:3, :3]
        pos_error = current_pos - goal[:3]
        R_goal = Rotation.from_rotvec(goal[3:]).as_matrix()
        orientation_error = -Rotation.from_matrix(R_goal @ R_curr.T).as_rotvec()
        d = np.concatenate([pos_error, orientation_error])
        if np.linalg.norm(d) < stopping_condition:
            print("end early")
            return thetas
        J_plus = np.linalg.pinv(J)
        cost_gradient = J_plus @ d
        thetas = thetas - step_size * cost_gradient
        thetas = np.clip(thetas, lower_limits, upper_limits)

    return thetas

def print_xyz_rotation(thetas, label=""):
    R = ee(thetas)[:3, :3]
    rx, ry, rz = Rotation.from_matrix(R).as_euler('xyz', degrees=False)
    print(f"{label} rotation about x, y, z (rad): [{rx:.3f}, {ry:.3f}, {rz:.3f}]")
    # print(f"{label} rotation about x, y, z (deg): [{np.degrees(rx):.1f}, {np.degrees(ry):.1f}, {np.degrees(rz):.1f}]")

def output_robot_curr_status(robot):
    print("Joint Values:")
    print(robot.joint_values)
    EE_transform = ee(robot.joint_values)
    print("End Effector FK:")
    print(EE_transform)
    print("Robot current pose:")
    curr_pose = current_pose(robot.joint_values)
    print(curr_pose)
    print_xyz_rotation(robot.joint_values)
    print()

def new_pose(robot, i, offset):
    # i :
    #   0 = translation along x axis
    #   1 = translation along y axis
    #   2 = translation along z axis
    #   3 = rotate about x axis
    #   4 = rotate about y axis
    #   5 = rotate about z axis
    # offset = distance to translation or theta to rotate
    #   provide distance in mm
    curr = current_pose(robot.joint_values)
    if i > 2:
        R_curr = ee(robot.joint_values)[:3, :3]
        axis = np.zeros(3)
        axis[i - 3] = 1
        R_goal = Rotation.from_rotvec(axis * offset) * Rotation.from_matrix(R_curr)
        return np.concatenate([curr[:3], R_goal.as_rotvec()])
    curr[i] += offset
    return curr



# detector = LightsaberDetector()
with Robot(visualize=True, open_browser=None) as robot:
    # input("Press enter to start free drive")
    # robot.free_drive()
    # input('Enter to stop\n')
    # robot.end_free_drive()
    output_robot_curr_status(robot)

    # print("Goal: +x 10cm")
    # # 10 cm = 100 mm
    # goal_pose = new_pose(robot, 0, 100)
    # print("Goal: ", goal_pose)
    # joint_updates = inverse_kinematics(robot.joint_values, goal_pose)
    # print("New theta", joint_updates)
    # robot.set_joint_targets(joint_updates)
    # output_robot_curr_status(robot)

    # print("Goal: +y 10cm")
    # goal_pose = new_pose(robot, 1, 100)
    # print("Goal: ",goal_pose)
    # joint_updates = inverse_kinematics(robot.joint_values, goal_pose)
    # print("New theta", joint_updates)
    # robot.set_joint_targets(joint_updates)
    # output_robot_curr_status(robot)

    # print("Goal: +z 10cm")
    # goal_pose = new_pose(robot, 2, 100)
    # print("Goal: ", goal_pose)
    # joint_updates = inverse_kinematics(robot.joint_values, goal_pose)
    # print("New theta",joint_updates)
    # robot.set_joint_targets(joint_updates)
    # output_robot_curr_status(robot)

    # print("Goal: +z pi/3")
    # goal_pose = new_pose(robot, 5, np.pi/3)
    # print("Goal: ", goal_pose)
    # joint_updates = inverse_kinematics(robot.joint_values, goal_pose)
    # print("New theta",joint_updates)
    # robot.set_joint_targets(joint_updates)
    # output_robot_curr_status(robot)

    input("Get Ready")
    z_offset = input("Enter distance to lightsaber (mm): ")
    goal_pose = new_pose(robot, 2, int(z_offset))
    print("Goal: ", goal_pose)
    joint_updates = inverse_kinematics(robot.joint_values, goal_pose)
    print("New theta",joint_updates)
    robot.set_joint_targets(joint_updates)
    output_robot_curr_status(robot)
    input("Press enter for gripper to close")
    robot.grip(0)
    input("Press enter to go up to original position")
    goal_pose = new_pose(robot, 2, -int(z_offset))
    print("Goal: ", goal_pose)
    joint_updates = inverse_kinematics(robot.joint_values, goal_pose)
    print("New theta",joint_updates)
    robot.set_joint_targets(joint_updates)
    output_robot_curr_status(robot)

    # robot.register_visualizer(detector)
    # robot.free_drive()
    # input('Enter to stop\n')
    # robot.end_free_drive()