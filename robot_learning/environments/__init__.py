"""
Define all environments and provide helper functions to load environments.
"""
import yaml
import os
import numpy as np
import gymnasium as gym


from InduMan.utils import get_task_cfg_path,get_assets_path

ROBOT_STATE_DIMS = {
    "ee_pose": 7,
    "gripper_position": 2,
    "joint_positions": 7,
    "joint_velocities": 7,
    "joint_forces": 54
}

OBSERVATION_DIMS = {
    "parts_state": 17,
    "agentview_img": (128, 128, 3),
    "hand_camera_img": (128, 128, 3),
    }

def make_env(name, config=None):
    """
    Creates a new environment instance with @name and @config.
    """
    if config.task_assets_path is None:
        config.task_assets_path = get_assets_path()

    env_cfg_path = os.path.join(get_task_cfg_path(), f"{name}.yaml")
    with open(env_cfg_path, 'r') as f:
        task_config = yaml.safe_load(f)

    if config.env_type == "assembly":
        from InduMan.scripts import FrankaAssemblyGym
        env = FrankaAssemblyGym(is_replay=False,
                        max_episode_steps=task_config['max_episode_steps'],
                      stage_units=task_config['stage_units'],
                      image_size=task_config['image_size'],
                      physics_dt=task_config['physics_dt'],
                      rendering_dt=task_config['rendering_dt'],
                      table_height=task_config['table_height'],
                      task_name=name,
                      task_message=task_config['task_message'],
                      objects_to_manipulate=task_config['objects_to_manipulate'],
                      objects_to_interact=task_config['objects_to_interact'],
                      support = task_config['support'],
                      task_assets_path = config.task_assets_path,
                      gripper_opened_position = task_config['gripper_opened_position'],
                      )
    elif config.env_type == "disassembly":
        from InduMan.scripts import FrankaDisassemblyGym
        env = FrankaDisassemblyGym(is_replay=False,
                                max_episode_steps=task_config['max_episode_steps'],
                                stage_units=task_config['stage_units'],
                                image_size=task_config['image_size'],
                                physics_dt=task_config['physics_dt'],
                                rendering_dt=task_config['rendering_dt'],
                                table_height=task_config['table_height'],
                                task_name=name,
                                task_message=task_config['task_message'],
                                objects_to_manipulate=task_config['objects_to_manipulate'],
                                objects_to_interact=task_config['objects_to_interact'],
                                task_assets_path=config.task_assets_path,
                                gripper_opened_position=task_config['gripper_opened_position'],
                                )
    else:
        raise ValueError(f"Unsupported task type: {config.env_type}")
    return env

def observation_space():   
    low, high = -np.inf, np.inf

    obs_dict = {}
    robot_state_dims = 0
    for dim_name, dim_size in ROBOT_STATE_DIMS.items():
        robot_state_dims += dim_size

    obs_dict["robot_state"] = gym.spaces.Box(low=low, high=high, shape=(robot_state_dims,), dtype=np.float32)

    for obs_name, dim in OBSERVATION_DIMS.items():
        if obs_name.endswith("img"):
            obs_dict[obs_name] = gym.spaces.Box(low=0, high=255, shape=dim, dtype=np.uint8)
        else:
            obs_dict[obs_name] = gym.spaces.Box(low=low, high=high, shape=(dim,), dtype=np.float32)

    return gym.spaces.Dict(obs_dict)

