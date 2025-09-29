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
from InduMan.utils import get_wdy_assets_path
from useful_scripts.bc_pipeline import BCPolicy, Standardizer,ActionStandardizer

# load task settings
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
                    gripper_opened_position=task_config['gripper_opened_position'],
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
                    support = task_config['support'],
                    task_assets_path = args.task_assets_path,
                    gripper_opened_position = task_config['gripper_opened_position'],
                    )
else:
    raise ValueError("Invalid task type")

# load bc model
# ======= 路径 =======
CKPT_PATH = "/home/wdy02/wdy_program/simulation_plus/IsaacLab/wdy_data/bc_data/bc_ckpts/ONE_PEG_IN_HOLE/best.pt"
SCALER_PATH = "/home/wdy02/wdy_program/simulation_plus/IsaacLab/wdy_data/bc_data/bc_ckpts/ONE_PEG_IN_HOLE/scaler.pkl"
ACTION_SCALER_PATH = "/home/wdy02/wdy_program/simulation_plus/IsaacLab/wdy_data/bc_data/bc_ckpts/ONE_PEG_IN_HOLE/action_scaler.pkl"

def _strip_module_prefix(state_dict):
    new = {}
    for k, v in state_dict.items():
        nk = k[len("module."):] if k.startswith("module.") else k
        new[nk] = v
    return new

def load_ckpt_and_build_model(ckpt_path: str, model_cls, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    # 支持 ckpt 直接是 state_dict / dict 包含 'model' / 'model_state_dict' / 'state_dict'
    state_dict = None
    if isinstance(ckpt, dict):
        state_dict = ckpt.get('model') or ckpt.get('model_state_dict') or ckpt.get('state_dict')
        model_kwargs = ckpt.get('model_kwargs', {}) or {}
    else:
        # ckpt 不是 dict，可能直接就是 state_dict
        state_dict = ckpt
        model_kwargs = {}

    # 如果 state_dict 有 module. 前缀，去掉
    if isinstance(state_dict, dict) and any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = _strip_module_prefix(state_dict)

    # 若 ckpt 中没有 model_kwargs，需要手工提供（可修改为你实际值）
    if not model_kwargs:
        raise RuntimeError("Checkpoint does not contain model_kwargs — please provide model_kwargs matching the training config.")

    # 确保 key 名与 BCPolicy 构造函数匹配：训练时你保存的是 input_dim/output_dim/hidden_fusion
    # BCPolicy 接受 input_dim, hidden_fusion, output_dim, dropout, pretrained_backbone 等
    # 若 ckpt 中用的是 output_dim，但你的 BCPolicy 使用 output_dim（如上），则直接传入
    model = model_cls(
        input_dim = model_kwargs.get('input_dim'),
        hidden_fusion = model_kwargs.get('hidden_fusion'),
        output_dim = model_kwargs.get('output_dim'),
        pretrained_backbone = model_kwargs.get('pretrained_backbone', True),
        dropout = model_kwargs.get('dropout', 0.0),
    ).to(device)

    if state_dict is None:
        print("No weights found in checkpoint; returning freshly initialized model.")
        return model, ckpt

    # 尝试只加载 shape 完全匹配的键；对 shape 不匹配的尝试做部分拷贝（拷贝重叠区域）
    model_state = model.state_dict()
    to_load = {}
    skipped = []
    partial = []
    for k_ck, v_ck in state_dict.items():
        if k_ck not in model_state:
            skipped.append(k_ck)
            continue
        v_model = model_state[k_ck]
        if v_ck.shape == v_model.shape:
            to_load[k_ck] = v_ck
        else:
            # 尝试部分拷贝到重叠 region（谨慎使用，可能破坏权重语义）
            try:
                min_shape = tuple(min(a, b) for a, b in zip(v_ck.shape, v_model.shape))
                if len(min_shape) == 0:
                    skipped.append(k_ck)
                    continue
                new = v_model.clone()
                slices = tuple(slice(0, m) for m in min_shape)
                new[slices] = v_ck[slices]
                to_load[k_ck] = new
                partial.append((k_ck, v_ck.shape, v_model.shape))
            except Exception:
                skipped.append(k_ck)

    # 报告信息
    print(f"Matched/partial keys: {len(to_load)}  Skipped keys: {len(skipped)}  Partial: {len(partial)}")
    if partial:
        print("Partial-loaded parameters (ckpt_shape -> model_shape):")
        for k, a, b in partial:
            print(f"  {k}: {a} -> {b}")
    if skipped:
        print("Skipped (not present in model):")
        for k in skipped[:20]:
            print("  ", k)
        if len(skipped) > 20:
            print("  ...")

    # 更新 model_state 并加载
    model_state.update(to_load)
    model.load_state_dict(model_state, strict=False)
    return model, ckpt

# 使用示例（替换你现有加载逻辑）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, ckpt = load_ckpt_and_build_model(CKPT_PATH, BCPolicy, device)
model.eval()
scaler = Standardizer.load(SCALER_PATH)
action_scaler = ActionStandardizer.load(ACTION_SCALER_PATH)

# ======= 创建环境 =======
env = my_env
obs = env.reset()

# 推荐：简单且高效
for k,v in obs.items():
    if k.endswith('_contact_forces'):
        contact_forces = v
        print('what is the contact forces:', contact_forces)

done = False
total_reward = 0
while simulation_app.is_running() and not done:
    env.world.step(render=True)  # 渲染环境
    if env.world.is_playing():

        # 1. 按训练时处理观测
        for k,v in obs.items():
            if k.endswith('_contact_forces'):
                contact_forces = v

        proprio_vec = np.concatenate([
            obs['ee_pose'],
            obs['joint_positions'],
            obs['joint_velocities'],
            obs['joint_forces_torques'],
            contact_forces
        ],axis=0)
        proprio_vec = torch.tensor(proprio_vec, dtype=torch.float32).unsqueeze(0).to(device)
        proprio_scaled = scaler.transform(proprio_vec)

        front_camera_rgb = torch.tensor(obs['agent_rgb'],dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2).to(device)
        hand_front_camera_rgb = torch.tensor(obs['hand_rgb'],dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2).to(device)

        # 2. 模型推理
        with torch.no_grad():
            pred = model(front_camera_rgb, hand_front_camera_rgb, proprio_scaled)
            action = action_scaler.inverse_transform(pred) # inverse transform for evaluation
            action = action.squeeze(0).cpu().numpy() # 转为 numpy array
        action = action*0.1  # 缩放动作
        # 3. 环境交互
        obs, reward, done, success, info = env.step(action)
        total_reward += reward

        # 可选：调试
        print(f"Step reward: {reward}, total: {total_reward}")

for i in range(100):
    env.world.step(render=True)  # 渲染环境
simulation_app.close()
print(f"Episode total reward: {total_reward}")
