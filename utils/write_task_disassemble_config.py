import os
import yaml
from utils import get_task_cfg_path

task_info = {
    "task_name": "DISASSEMBLE_3PIN_PLUG_SOEKCT",
    "task_message": ["Disassemble the 3pin_plug in the middle position of socket smoothly.", 
                     "Remove the 3pin_plug in the middle position of socket sleeve.",
                     "Detach the 3pin_plug in the middle position of socket."],
    "task_setting_suggestion":"This task aims to disassemble the 3pin_plug in the middle position of socket.",
    "physics_dt": 120,
    "rendering_dt": 120,
    "stage_units": 1.0, #  in meters
    "max_episode_steps":1000,
    "image_size": [128, 128],
    "stand_position": [[0.0, 0.0, 0.775]],
    "robot": {
        "joint_positions": [0.09013208, -0.526969 ,  -0.09634338, -2.5128212,  -0.04378112,  1.9551778, 0.8178173],
        "gripper_state": "open",
    },
    "gripper_opened_position": [0.05, 0.05],
    "table_height": 1.03,
    "objects_to_manipulate":{
        "object_1":{
            "name": "plug",
            "mass": 0.001,
            "usd_path": "/task_assets/PLUGS_SOCKETS/plug_3pin.usdc",
            "interact_object":"socket",
            "offset_with_interact_obj": [[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]],
            "target_area":[0.6,-0.2],
            "success_criteria": [[0.1,0.1,0.003], #success criteria for position error in meters
                                 [0.0001]], #success criteria for force error in Newtons
            },
    },

    "objects_to_interact":{
        "object_1":{
            "name": "socket",
            "area": [[0.6,0.1],[0.5,0.005]],
            "offset": 0.0, # offset in meters, distance from next object in x axis
            "height":0.001,  # Height in meters, distance from the table surface
            "orientation": [[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]],  # euler angle in degrees
            "mass": 100.0,  # 100 kg for hole, rigidprim can not be fixed, so we set a large mass
            "usd_path": "/task_assets/PLUGS_SOCKETS/sockets.usdc",
            "track_contact_forces": True,
            },
    },
}

task_info_path = os.path.join(get_task_cfg_path(), f"{task_info['task_name']}.yaml")
os.makedirs(os.path.dirname(task_info_path), exist_ok=True)

with open(task_info_path, "w") as f:
    yaml.dump(task_info, f, allow_unicode=True,sort_keys=False,indent = 4)
print(f"Task info saved to {task_info_path}")
