"""
    This script is used to collect data when replaying trajectory data in a json file, and store them in a hdf5 file.
    But it is for disassemble tasks only, not for assemble tasks.
"""

import numpy as np
import h5py
import os
import sys
import yaml

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.rotations import rot_matrix_to_quat

# sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from InduMan.scripts import FrankaDisassemblyGym
from InduMan.utils import get_assets_path, get_task_cfg_path, get_default_data_path


def parser_args():
    import argparse
    parser = argparse.ArgumentParser(description="Franka Gym Environment")
    parser.add_argument('--task_name', type=str, default='RECTANGLE_PEG_OUT_HOLE', help='Name of the task to run')
    parser.add_argument('--task_assets_path', type=str, default=get_assets_path(), help='Path to the task assets')
    parser.add_argument('--data_base_path', default=None, help='Path to the json files logged by the task')
    parser.add_argument('--use_depth', type=bool, default=False, help='Whether to use depth camera')
    parser.add_argument('--use_hand_back_camera',action='store_true', help='Whether to use hand back camera')
    args = parser.parse_args()
    # select assets path, convenient for running in different environments
    print("\033[94mChoose the assets path for the task:\033[0m") 
    print("\033[93m1. use assets in program folder\033[0m")
    print("\033[92m2. use assets in isaac assets folder\033[0m")
    choice = input("\033[96mChoose a number (1/2): \033[0m").strip() 
    if choice == '1':
        args.task_assets_path = get_assets_path()
    elif choice == '2':
        args.task_assets_path = None
    else:
        print("Invalid input, default to use isaac assets folder.")
        args.task_assets_path = None
 
    return args

# extract and sort timestamps
def extract_timestamp(filename):
    try:
        # split the string by the underscore character ('_'), and the first segment is the timestamp
        ts_str = filename.split('_')[0]
        return float(ts_str)
    except (ValueError, IndexError):
        # if the timestamp cannot be parsed, assign a maximum value to ensure it is sorted to the end
        return float('inf')
    
def main(args=None):
    # load task settings
    task_name = args.task_name
    task_config_path = os.path.join(get_task_cfg_path(), f"{task_name}.yaml")
    with open(task_config_path, 'r') as f:
        task_config = yaml.safe_load(f)
    
    # create environment
    objects_to_manipulate_dict = task_config['objects_to_manipulate']
    objects_to_interact_dict = task_config['objects_to_interact']
    my_env = FrankaDisassemblyGym(is_replay=False,
                        max_episode_steps=task_config['max_episode_steps'],
                      stage_units=task_config['stage_units'],
                      image_size=task_config['image_size'],
                      physics_dt=task_config['physics_dt'],
                      rendering_dt=task_config['rendering_dt'],
                      table_height=task_config['table_height'],
                      task_name=task_name,
                      task_message=task_config['task_message'],
                      objects_to_manipulate=objects_to_manipulate_dict,
                      objects_to_interact=objects_to_interact_dict,
                      task_assets_path=args.task_assets_path,
                      )
    # get data files
    data_base_path = os.path.join(get_default_data_path(), task_name) if args.data_base_path is None else args.data_base_path
    h5file_path = os.path.join(data_base_path,f'h5_file/{task_name}.h5')
    os.makedirs(os.path.dirname(h5file_path), exist_ok=True)

    # retrieve all file names (only the names, not the full paths)
    files = [
        f for f in os.listdir(data_base_path)
        if os.path.isfile(os.path.join(data_base_path, f))
    ]
    # the sorted list of file names
    sorted_files = sorted(files, key=extract_timestamp)
    # build the complete file path, follow the sorted order(by timestamp)
    data_file_paths = [os.path.join(data_base_path, f) for f in sorted_files]

    # create an HDF5 file
    hdf5_file = h5py.File(h5file_path, 'w')
    # create a group for storing all data frames
    group = hdf5_file.create_group("data_frames")

    group.attrs['env_name'] = 'FrankaGym'
    group.attrs['task_name'] = task_name

    # ready data
    data_logger = my_env.data_logger
    # make sure tool_center_marker and hand_marker can not be captured by camera
    my_env.tool_center_marker.set_visibility(False)
    my_env.hand_marker.set_visibility(False)

    franka_metadata = my_env.franka._articulation_view._metadata # get franka metadata for joint names and indices
    forces_joint_indices = [1 + franka_metadata.joint_indices[name] for name in my_env.franka.dof_names]

    use_depth = args.use_depth
    success_demo_count = 0
    failed_files = [] # log the failed files
    for i in range(len(data_file_paths)):
        # load data from json file
        data_logger.load(data_file_paths[i])
        demo_name = os.path.basename(data_file_paths[i])
        print(f"Processing demo {demo_name}...")

        task_objects_dict = my_env.objects_of_task # get all task objects with their names in a dict

        # store data for every demo
        episode_step = []
        is_success = []
        actions = []
        rewards = []    
        objects_pose_dict = {}
        for obj_name, obj in task_objects_dict.items(): # get all objects position with names
            objects_pose_dict[obj_name] = []
        robot_joint_states = []
        gripper_joint_states = []
        robot_measured_joint_forces = []
        objects_force_dict = {} # use a dict to store all objects force
        for obj_key, obj_info in objects_to_manipulate_dict.items(): # get all manipulate objects with their interact objects
            key = f"{obj_info['name']}_{obj_info['interact_object']}_forces"
            objects_force_dict[key] = []
        # store observations
        agentview_rgb =[]
        agentview_depth = []
        hand_front_camera_rgb = []
        hand_front_camera_depth = []
        hand_back_camera_rgb =[]
        hand_back_camera_depth = []
        ee_poses = []

        # every data file has a different init state for objects
        for obj_name, obj in task_objects_dict.items(): # set objects init position and orientation
            position = data_logger.get_data_frame(data_frame_index=0).data['objects'][obj_name]['position']
            rotation = data_logger.get_data_frame(data_frame_index=0).data['objects'][obj_name]['orientation']
            obj.set_world_poses(positions= position, orientations=rotation)
        
        last_action_step = data_logger.get_data_frame(data_frame_index=0).data['episode_steps']-1 # make a last action
        first_action = False
        while simulation_app.is_running():
            my_env.world.step(render=True)

            if my_env.world.is_playing():
                if my_env.world.current_time_step_index <= data_logger.get_data_frame(data_logger.get_num_of_data_frames()-1).current_time_step:
                    data_frame_first = data_logger.get_data_frame(data_frame_index = 0)
                    #print("current_time_step_index: ",my_env.world.current_time_step_index,"last_time_step: ",data_frame_first.current_time_step)

                    if my_env.world.current_time_step_index == data_frame_first.current_time_step:
                        first_action = True
                    
                    if first_action:
                        # print("Running demo...")
                        data_frame_index = my_env.world.current_time_step_index - data_frame_first.current_time_step
                        data_frame = data_logger.get_data_frame(data_frame_index=data_frame_index)
                        # get last action step
                        current_action_step = data_frame.data['episode_steps']
                    
                        # set robot joint position and gripper state
                        action = ArticulationAction(joint_positions=data_frame.data['robot_states']['joint_positions'][:7],
                                                    joint_indices=list(range(7)),)
                        #print("current_action_step: ",current_action_step,"robot action:",action)
                        my_env.articulation_controller.apply_action(control_actions= action)
                        if data_frame.data['actions'][6] == 0:
                            # make sure gripper only open once
                            if np.all(my_env.gripper.get_joint_positions() >= my_env.gripper.joint_opened_positions):
                                pass
                            elif np.all(my_env.gripper.get_joint_positions() < my_env.gripper.joint_opened_positions- 0.001):
                                my_env.gripper.open()
                        else:
                            my_env.gripper.close()
                        
                        my_env.is_done() # update is_success flag
                    
                        if current_action_step - last_action_step == 1: # to insure log action step state
                            episode_step.append(data_frame.data['episode_steps'])
                            is_success.append(my_env.is_success)
                            actions.append(data_frame.data['actions'])
                            reward = 1.0 if my_env.is_success else 0.0
                            rewards.append(reward)
                            robot_joint_states.append(np.concatenate([my_env.franka.get_joints_state().positions[:7], my_env.franka.get_joints_state().velocities[:7]])) # only store robot joint not gripper
                            gripper_joint_states.append(my_env.gripper.get_joint_positions())
                            robot_measured_joint_forces.append(my_env.franka.get_measured_joint_forces(forces_joint_indices).flatten())
                            
                            for obj_name, obj in task_objects_dict.items(): # get objects poses
                                pos,ori = obj.get_world_poses()
                                objects_pose_dict[obj_name].append(np.concatenate([pos[0],ori[0]]))
                            for obj_key, obj_info in objects_to_manipulate_dict.items():
                                key = f"{obj_info['name']}_{obj_info['interact_object']}_forces"
                                force = task_objects_dict[obj_info['name']].get_contact_force_matrix(None,dt=1./60)[0][0]
                                objects_force_dict[key].append(force)
                            
                            # store observations
                            agentview_rgb.append(my_env.get_camera_observation(my_env.camera_world))
                            hand_front_camera_rgb.append(my_env.get_camera_observation(my_env.camera_hand))
                            if args.use_hand_back_camera:
                                        hand_back_camera_rgb.append(my_env.get_camera_observation(my_env.camera_hand_back))
                            ee_positioin,ee_ori_mat = my_env.controller.compute_end_effector_pose()
                            ee_ori_quat = rot_matrix_to_quat(ee_ori_mat)
                            ee_poses.append(np.concatenate([ee_positioin,ee_ori_quat]))
                            if use_depth :
                                agentview_depth.append(my_env.get_camera_observation(my_env.camera_world, depth=True))
                                hand_front_camera_depth.append(my_env.get_camera_observation(my_env.camera_hand, depth=True))
                                if args.use_hand_back_camera:
                                    hand_back_camera_depth.append(my_env.get_camera_observation(my_env.camera_hand_back, depth=True))
                            last_action_step = current_action_step
                else:
                    print("Task success? ",my_env.is_success)
                    # store data for the successful state
                    episode_step.append((data_frame.data['episode_steps'] + 1))
                    is_success.append(my_env.is_success)
                    actions.append(data_frame.data['actions'])
                    reward = 1.0 if my_env.is_success else 0.0
                    rewards.append(reward)
                    robot_joint_states.append(np.concatenate([my_env.franka.get_joints_state().positions[:7], my_env.franka.get_joints_state().velocities[:7]])) # only store robot joint not gripper
                    gripper_joint_states.append(my_env.gripper.get_joint_positions())
                    robot_measured_joint_forces.append(my_env.franka.get_measured_joint_forces(forces_joint_indices).flatten())
                    
                    for obj_name, obj in task_objects_dict.items(): # get objects poses
                        pos,ori = obj.get_world_poses()
                        objects_pose_dict[obj_name].append(np.concatenate([pos[0],ori[0]]))
                    for obj_key, obj_info in objects_to_manipulate_dict.items():
                        key = f"{obj_info['name']}_{obj_info['interact_object']}_forces"
                        force = task_objects_dict[obj_info['name']].get_contact_force_matrix(None,dt=1./60)[0][0]
                        objects_force_dict[key].append(force)
                    
                    # store observations
                    agentview_rgb.append(my_env.get_camera_observation(my_env.camera_world))
                    hand_front_camera_rgb.append(my_env.get_camera_observation(my_env.camera_hand))
                    if args.use_hand_back_camera:
                        hand_back_camera_rgb.append(my_env.get_camera_observation(my_env.camera_hand_back))
                    ee_positioin,ee_ori_mat = my_env.controller.compute_end_effector_pose()
                    ee_ori_quat = rot_matrix_to_quat(ee_ori_mat)
                    ee_poses.append(np.concatenate([ee_positioin,ee_ori_quat]))
                    if use_depth :
                        agentview_depth.append(my_env.get_camera_observation(my_env.camera_world, depth=True))
                        hand_front_camera_depth.append(my_env.get_camera_observation(my_env.camera_hand, depth=True))
                        if args.use_hand_back_camera:
                            hand_back_camera_depth.append(my_env.get_camera_observation(my_env.camera_hand_back, depth=True))
                    break
        if my_env.is_success: # store data only when task is success
            # create demo group
            demo_group = group.create_group(f"demo_{success_demo_count}")
            success_demo_count += 1
            # create observations group
            obs_demo_group = demo_group.create_group("observations")
            # store data in hdf5 file when task is success
            # store data file name for every demo
            demo_group.attrs['demo_name'] = demo_name
            
            # store observations
            demo_group.create_dataset("episode_step", data=episode_step)
            demo_group.create_dataset("is_success", data=is_success)
            demo_group.create_dataset("actions", data=actions)
            demo_group.create_dataset("rewards", data=rewards)
            demo_group.create_dataset("robot_joint_states", data=robot_joint_states)
            demo_group.create_dataset("gripper_joint_states", data=gripper_joint_states)
            demo_group.create_dataset("robot_measured_joint_forces",data=robot_measured_joint_forces)
            for obj_name, obj_pose in objects_pose_dict.items():
                demo_group.create_dataset(f"{obj_name}_poses", data=obj_pose)
            
            for obj_name, obj_force in objects_force_dict.items():
                demo_group.create_dataset(obj_name, data=obj_force)

            obs_demo_group.create_dataset("agentview_rgb", data=agentview_rgb)
            obs_demo_group.create_dataset("hand_camera_rgb", data=hand_front_camera_rgb)
            if args.use_hand_back_camera:
                obs_demo_group.create_dataset("hand_back_camera_rgb", data=hand_back_camera_rgb)
            obs_demo_group.create_dataset("ee_pos", data=ee_poses)
            if use_depth:
                obs_demo_group.create_dataset("agentview_depth", data=agentview_depth)
                obs_demo_group.create_dataset("hand_front_camera_depth", data=hand_front_camera_depth)
                if args.use_hand_back_camera:
                    obs_demo_group.create_dataset("hand_back_camera_depth",data=hand_back_camera_depth) 
            # flush to release memory
            hdf5_file.flush()
        else:
            # log the failed files
            failed_files.append(demo_name)
        my_env.reset()

    print("success_demo_count: ",success_demo_count, "Failed files: ",failed_files)
    hdf5_file.close()
    simulation_app.close()

if __name__ == '__main__':
    args = parser_args()
    main(args)