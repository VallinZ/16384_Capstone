from xarm7_lib import Robot
from libcapstone import LightsaberDetector
import time
import numpy as np
from scipy.spatial.transform import Rotation

np.set_printoptions(precision=2, suppress=True)

link_length = [0, 0, 0, 52.5, 77.5, 0, 76]
link_twist = [0, -np.pi/2, np.pi/2, np.pi/2, np.pi/2, np.pi/2, -np.pi/2]
link_offset = [267, 0, 293, 0, 342.5, 0, 269]
lower_limits = np.array([-2*np.pi, -2.059, -2*np.pi, -0.1919, -2*np.pi, -1.6929, -2*np.pi])
upper_limits = np.array([2*np.pi,  2.0944,  2*np.pi,  3.9270,  2*np.pi,  3.1415,  2*np.pi])
COLOR = "red"

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

def get_position(T):
    return T[:3, -1]

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

# given position and rotation matrix
def new_pose_v2(pos, rot):
    # 1000 mm = 1 m
    rotvec = Rotation.from_matrix(rot).as_rotvec()
    return np.concatenate([pos, rotvec])

def closer_end(robot, ends):
    robot_pos = ee(robot.joint_values)[:3, -1]
    distance0 = np.linalg.norm(robot_pos - ends[0])
    distance1 = np.linalg.norm(robot_pos - ends[1])
    if distance0 < distance1:
        return ends[0]
    else:
        return ends[1]

def new_tip_pos(pos, i, offset):
    # i :
    #   0 = translation along x axis
    #   1 = translation along y axis
    #   2 = translation along z axis
    # offset = distance to translation or theta to rotate
    #   provide distance in mm
    new_pos = pos.copy()
    new_pos[i] += offset
    return new_pos

def ordered_ends(robot, ends):
    # return [handle end, tip end]
    robot_pos = ee(robot.joint_values)[:3, -1]
    distance0 = np.linalg.norm(robot_pos - ends[0])
    distance1 = np.linalg.norm(robot_pos - ends[1])
    if distance0 < distance1:
        return [ends[0], ends[1]]
    else:
        return [ends[1], ends[0]]

def tip_T(robot):
    ee_T = ee(robot.joint_values)
    y_ee = ee_T[:3, 1]
    T_list = []
    for _ in range(10):
        ends = det.query("red") # Change to whatever color lightstick
        if ends is None or np.shape(ends) != (2, 3):
            continue
        ends = np.asarray(ends) * 1000
        ends = ordered_ends(robot, ends)
        tip_end = ends[1]
        handle_end = ends[0]
        z = tip_end - handle_end
        z /= np.linalg.norm(z)
        y = y_ee - np.dot(y_ee, z) * z
        y /= np.linalg.norm(y)
        x = np.cross(y,z)
        tipT = np.eye(4)
        tipT[:3, :3] = np.column_stack((x, y, z))
        tipT[:3, 3] = tip_end
        T = np.linalg.inv(ee_T) @ tipT
        T_list.append(T)
    T_stack = np.asarray(T_list)
    avg_T = np.eye(4)
    avg_T[:3, :3] = Rotation.from_matrix(T_stack[:, :3, :3]).mean().as_matrix()
    avg_T[:3, 3] = T_stack[:, :3, 3].mean(axis=0)
    return avg_T

# Convert to a gripper pose: T_base→ee = T_base→tip_desired · inverse(T_ee→tip).
def tip_to_ee(desired_tip_T, ee_tip_T):
    base_ee_T = desired_tip_T @ np.linalg.inv(ee_tip_T)
    return base_ee_T

# T_base→tip_current = T_base→ee(current joints) · T_ee→tip
def curr_tip_T(robot, ee_tip_T):
    ee_T = ee(robot.joint_values)
    current_tip_T = ee_T @ ee_tip_T
    return current_tip_T

def curr_tip_pos(robot, ee_tip_T):
    current_tip_T = curr_tip_T(robot, ee_tip_T)
    current_tip_pos = get_position(current_tip_T)
    return current_tip_pos

def fit_plane(points):
    P = np.asarray(points)                 # (N, 3)
    c = P.mean(axis=0)                     # centroid
    P_centered = P - c
    _, S, Vt = np.linalg.svd(P_centered, full_matrices=False)
    n = Vt[-1]                             # direction of least variance (unit length)
    d = np.dot(n, c)                       # plane: n . x = d
    return n, d, c, S

det = LightsaberDetector()

# Uncomment the line below for real robot
# with Robot() as robot:
with Robot(visualize=True, open_browser=None) as robot:
    grip = int(input("Enter 1 to open grip or 0 to close grip: "))
    robot.grip(grip)
    # input("Press enter to start free drive")
    # robot.free_drive()
    # input('Enter to stop\n')
    # robot.end_free_drive()
    output_robot_curr_status(robot)
    robot.register_visualizer(det)

    part = int(input("Enter what part you want to do: "))
    if part == 1:
        input("Press enter to get started")
        print("Pos1")
        # 10 cm = 100 mm

        pos1 = np.array([500, 0, 0])
        rot1 = np.array([[1, 0, 0],
                        [0, -1, 0],
                        [0, 0, -1]])
        goal_pose = new_pose_v2(pos1, rot1)
        print("Goal: ", goal_pose)
        joint_updates = inverse_kinematics(robot.joint_values, goal_pose)
        print("New theta", joint_updates)
        robot.set_joint_targets(joint_updates)
        output_robot_curr_status(robot)
        input("Press enter for the next pose")

        print("Pos2")
        pos2 = np.array([600, 0, 300])
        rot2 = np.array([[0, 0, 1],
                            [0, -1, 0],
                            [1, 0, 0]])
        goal_pose = new_pose_v2(pos2, rot2)
        print("Goal: ",goal_pose)
        joint_updates = inverse_kinematics(robot.joint_values, goal_pose)
        print("New theta", joint_updates)
        robot.set_joint_targets(joint_updates)
        output_robot_curr_status(robot)
        input("Press enter for the next pose")

        print("Pos3")
        pos3 = np.array([400, 200, 250])
        rot3 = np.array([[0, 1, 0],
                            [0, 0, 1],
                            [1, 0, 0]])
        goal_pose = new_pose_v2(pos3, rot3)
        print("Goal: ", goal_pose)
        joint_updates = inverse_kinematics(robot.joint_values, goal_pose)
        print("New theta",joint_updates)
        robot.set_joint_targets(joint_updates)
        output_robot_curr_status(robot)

    if part == 2:
        input("Press enter to get started")
        grab_type = input("Enter 1 for manual or 2 for auto: ")
        if grab_type == "1":
            input("Press enter to get started")
            current_pos = robot.joint_values
            valid = False
            while not valid:
                input("Press enter to move robot close to lightstick (gripper right above handle)")
                robot.free_drive()
                input('Enter to stop\n')
                robot.end_free_drive()
                output_robot_curr_status(robot)
                z_offset = input("Enter distance to lightsaber (mm): ")
                goal_pose = new_pose(robot, 2, -1*int(z_offset))
                print("Goal: ", goal_pose)
                joint_updates = inverse_kinematics(robot.joint_values, goal_pose)
                print("New theta",joint_updates)
                robot.set_joint_targets(joint_updates)
                output_robot_curr_status(robot)
                temp_check = input("Enter 1 to confirm gripper location or enter 2 to restart: ")
                if int(temp_check) == 1:
                    valid = True
                else:
                    print("Returning robot to starting location")
                    robot.set_joint_targets(current_pos)
                    output_robot_curr_status(robot)
        if grab_type == "2":
            input("Press enter to move robot close to lightstick")
            robot.free_drive()
            input('Enter to stop\n')
            robot.end_free_drive()
            output_robot_curr_status(robot)        
            samples = []
            for _ in range(10):
                ends = det.query(COLOR) # Change to whatever color lightstick
                tip_end = closer_end(robot, ends)
                samples.append(tip_end)
            avg_tip_end = np.mean(samples, axis=0)*1000
            print(avg_tip_end)
            rot = np.array([[1, 0, 0],
                            [0, -1, 0],
                            [0, 0, -1]])
            goal_pose = new_pose_v2(avg_tip_end, rot)
            print("Goal: ", goal_pose)
            joint_updates = inverse_kinematics(robot.joint_values, goal_pose)
            print("New theta", joint_updates)
            input("Press enter to move to lightsaber")
            robot.set_joint_targets(joint_updates)
            output_robot_curr_status(robot)
        input("Press enter for gripper to close")
        robot.grip(0)
        z_up = input("Enter the amount in mm to go up")
        goal_pose = new_pose(robot, 2, int(z_up))
        print("Goal: ", goal_pose)
        joint_updates = inverse_kinematics(robot.joint_values, goal_pose)
        print("New theta",joint_updates)
        robot.set_joint_targets(joint_updates)
        output_robot_curr_status(robot)
        ee_tip_T = tip_T(robot)
        current_tip_pos = curr_tip_pos(robot, ee_tip_T)
        y_offset = -1*int(input("Enter the amount to go right horizontally (mm): "))
        new_tip_position = new_tip_pos(current_tip_pos, 1, y_offset)
        desired_tip_T = curr_tip_T(robot, ee_tip_T).copy()
        desired_tip_T[:3, 3] = new_tip_position  
        base_ee_T = tip_to_ee(desired_tip_T, ee_tip_T)
        new_ee_pose = current_pose(base_ee_T)
        print("Goal: ", new_ee_pose)
        joint_updates = inverse_kinematics(robot.joint_values, new_ee_pose)
        print("New theta",joint_updates)
        robot.set_joint_targets(joint_updates)
        output_robot_curr_status(robot)
        # Holding up straight
        input("Press enter to continue to holding the lightsaber upstraight")
        tip_T_curr = curr_tip_T(robot, ee_tip_T)
        z_now = tip_T_curr[:3, 2]
        z_now /= np.linalg.norm(z_now)
        R_align,_ = Rotation.align_vectors([[0,0,1]], [z_now])
        R_current = tip_T_curr[:3, :3]
        desired_rot = R_align.as_matrix() @ R_current
        desired_tip_T = tip_T_curr.copy()
        desired_tip_T[:3, :3] = desired_rot
        base_ee_T = tip_to_ee(desired_tip_T, ee_tip_T)
        new_ee_pose = current_pose(base_ee_T)
        print("Goal: ", new_ee_pose)
        joint_updates = inverse_kinematics(robot.joint_values, new_ee_pose)
        print("New theta",joint_updates)
        robot.set_joint_targets(joint_updates)
        output_robot_curr_status(robot)
    if part == 3:
        input("Press enter to get started")
        input("Hold lightsaber in initial position")
        # tip, then hilt
        ends=det.query(COLOR)
        print(ends)
        input("Sweep lightsaber in one direction to confirm end indices")
        ends=det.query(COLOR)
        print(ends)
        ind_confirm = int(input("Enter 0 if tip is at index 0 and handle at index 1, else enter 1: "))
        tip_index = 0
        if ind_confirm == 1:
            tip_index = 1
        duration = 2.5  # Choose any duration in the 2-3 second range, e.g., 2.0 to 3.0
        start_time = time.perf_counter()
        tip_points = []
        while (time.perf_counter() - start_time) < duration:
            ends=det.query(COLOR)
            if ends is None or np.shape(ends) != (2, 3):
                continue 
            tip_points.append(ends[tip_index])
        n, d, c, S = fit_plane(tip_points)         # tip_points is (N, 3)
        print("normal:", n, "d:", d)
        print("singular values:", S)
        plane = (n[0], n[1], n[2], -d)
        det.draw_plane(plane, center=c)

    
# Convert to a gripper pose: T_base→ee = T_base→tip_desired · inverse(T_ee→tip).
# Convert that pose to the 6D form your IK expects (and mm if you’re in mm).