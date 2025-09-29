import torch
import pickle
import numpy as np
import gym
import json
import yaml
import os
import sys
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--task_name", type=str, default="ONE_PEG_IN_HOLE", help="Task name")
parser.add_argument("--task_type",type=str, default="assemble", help="Task type(disassemble/assemble)")
parser.add_argument("--task_assets_path", type=str, default=None, help="Path to task assets")
args = parser.parse_args()

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from useful_scripts.iql_pipeline import IQLAgent
from InduMan.utils import get_wdy_assets_path

def preprocess_obs(obs):
    """把环境的obs转成 agent 需要的格式"""
    obs_img_keys = ['agent_rgb', 'hand_rgb']
    agent_img_keys =["agentview_rgb","hand_camera_rgb"]
    robot_joint_state = np.concatenate([obs["joint_positions"], obs["joint_velocities"]])
    vec = np.concatenate([obs['ee_pose'], obs['hole_pose'],obs['peg_pose'], 
                          robot_joint_state, obs['joint_forces_torques'], 
                          obs['gripper_state'], obs['peg_contact_forces']],axis=0)
    vec = torch.tensor(vec, dtype=torch.float32, device=device)
    imgs = {}
    for k in obs_img_keys:
        if k in obs:
            img = np.asarray(obs[k])
            if img.ndim == 3 and img.shape[2] in (1,3):  # HWC -> CHW
                t = torch.tensor(img.transpose(2,0,1), dtype=torch.float32, device=device).unsqueeze(0)
            else:
                t = torch.tensor(img, dtype=torch.float32, device=device).unsqueeze(0)
            imgs[k] = t
    # map obs keys -> agent expected keys
    mapped_imgs = {}
    # 若 agent 期望 "agentview_rgb","hand_camera_rgb"，就做映射
    if 'agent_rgb' in imgs:
        mapped_imgs['agentview_rgb'] = imgs['agent_rgb']
    if 'hand_rgb' in imgs:
        mapped_imgs['hand_camera_rgb'] = imgs['hand_rgb']

    return vec, mapped_imgs

def fix_state_dict_keys(sd: dict):
    new_sd = {}
    for k, v in sd.items():
        nk = k
        # remove dataparallel/module prefixes
        nk = nk.replace(".module.", ".")
        nk = nk.replace("module.", "")
        # if proj was saved at encoders.<name>.proj -> move under .encoder.proj
        # e.g. encoders.agentview_rgb.proj.weight -> encoders.agentview_rgb.encoder.proj.weight
        parts = nk.split(".")
        if len(parts) >= 3 and parts[0] == "encoders" and parts[2] == "proj":
            # rebuild key inserting 'encoder' after encoder name
            nk = ".".join([parts[0], parts[1], "encoder"] + parts[2:])
        new_sd[nk] = v
    return new_sd

# ====== 环境 ======
# load task settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
task_name = args.task_name
task_config_path = os.path.join(os.getcwd(), "wdy_file/task_config", f"{task_name}.yaml")
with open(task_config_path, 'r') as f:
    task_config = yaml.safe_load(f)

if args.task_type == "disassemble":
    from wdy_file.wdy_disassemble_gym import FrankaGym
    my_env = FrankaGym(is_replay=False,
                        max_episode_steps=task_config['max_episode_steps'],
                    stage_units=task_config['stage_units'],
                    image_size=task_config['image_size'],
                    physics_dt=task_config['physics_dt'],
                    rendering_dt=task_config['rendering_dt'],
                    table_height=task_config['table_height'],
                    task_name=task_name,
                    task_message=task_config['task_message'],
                    objects_to_manipulate=task_config['objects_to_manipulate'],
                    objects_to_interact=task_config['objects_to_interact'],
                    task_assets_path=args.task_assets_path,
                    )
elif args.task_type == "assemble":
    from wdy_file.wdy_assemble_gym import FrankaGym
    my_env = FrankaGym(is_replay=False,
                    max_episode_steps=task_config['max_episode_steps'],
                    stage_units=task_config['stage_units'],
                    image_size=task_config['image_size'],
                    physics_dt=task_config['physics_dt'],
                    rendering_dt=task_config['rendering_dt'],
                    table_height=task_config['table_height'],
                    task_name=task_name,
                    task_message=task_config['task_message'],
                    objects_to_manipulate=task_config['objects_to_manipulate'],
                    objects_to_interact=task_config['objects_to_interact'],
                    support=task_config['support'],
                    task_assets_path=args.task_assets_path,
                    )
else:
    raise ValueError("Invalid task type")

obs = my_env.reset()
vec, imgs = preprocess_obs(obs)
obs_vec_dim = vec.numel()

# ====== 加载模型 ======
agent = IQLAgent(
    obs_vec_dim=obs_vec_dim,     # 根据环境实际维度改
    act_dim=7,          # 动作维度
    r3m=True,
    img_keys=["agentview_rgb","hand_camera_rgb"],   # 你的相机 key
)
sd = torch.load("/home/wdy02/wdy_program/simulation_plus/IsaacLab/wdy_data/iql_data/iql_ckpts/best_model.pth", map_location=device)

fixed_sd = fix_state_dict_keys(sd)
# 尝试严格加载；若失败，再用 strict=False
try:
    agent.load_state_dict(fixed_sd, strict=True)
except RuntimeError as e:
    print("Strict load failed, will try non-strict load:", e)
    missing, unexpected = agent.load_state_dict(fixed_sd, strict=False)
    print("Loaded with strict=False. Missing keys:", missing)
    print("Unexpected keys:", unexpected)
agent.to(device)
agent.eval()

# ====== 测试推理 ======
done, total_reward = False, 0

while simulation_app.is_running and not done:
    my_env.world.step(render=True)  # 让环境运行一帧
    if my_env.world.is_playing():

        with torch.no_grad():
            x = agent.make_repr(vec, imgs)  # 获取 agent 的表示
            action = agent.actor(x)  # 获取动作
        
        action = action.cpu().numpy()[0]    # 转 numpy 给环境执行

        print(f"action: {action}","type:",type(action))
        if np.any(action > 0.01):
            print("Warning: action out of range, clip to 0.1")
            action = action * 0.01 # 根据实际动作范围归一化
        obs, reward, done, _, _ = my_env.step(action)
        vec, imgs = preprocess_obs(obs)

        total_reward += reward

    print(f"Total reward = {total_reward}")

simulation_app.close()
