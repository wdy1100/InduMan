# h5_to_npz_single.py
import os
import argparse
import h5py
import numpy as np

def _has(obj, key):
    try:
        obj[key]
        return True
    except KeyError:
        return False

def convert_one_h5(h5_path: str, out_dir: str, per_timestep: bool = False):
    os.makedirs(out_dir, exist_ok=True)
    with h5py.File(h5_path, "r") as f:
        if not _has(f, "/data_frames"):
            raise KeyError("H5 内没有 '/data_frames' 分组")
        df = f["/data_frames"]

        # 找到所有 demo_* 组
        demo_keys = sorted([k for k in df.keys() if k.startswith("demo_")])
        if not demo_keys:
            raise RuntimeError("未在 /data_frames 下找到任何 demo_* 分组")

        for demo_key in demo_keys:
            g = df[demo_key]
            print(f"[Convert] /data_frames/{demo_key}")

            # ------- 路径映射（按你给的结构）-------
            # 图像
            front_rgb = g["observations/agentview_rgb"][()]          # (T,128,128,3) uint8
            wrist_rgb = g["observations/hand_camera_rgb"][()]        # (T,128,128,3) uint8
            # 末端位姿（注意你的数据是7维，可能是 pos(3)+quat(4)；BC 代码会自适应维度）
            ee_pose   = g["observations/ee_poses"][()]               # (T,7) float64
            # 力信息
            joint_forces = g["robot_measured_joint_forces"][()]      # (T,54) float32
            # 某些数据集把接触力放在根；若你之后有别的命名可在此扩展
            if _has(g, "peg_hole_forces"):
                contact_forces = g["peg_hole_forces"][()]            # (T,3) float32
            elif _has(g, "observations/peg_hole_forces"):
                contact_forces = g["observations/peg_hole_forces"][()]
            else:
                raise KeyError("未找到 peg_hole_forces(根或 observations 下）")
            # robot state
            joint_states = g["robot_joint_states"][()]               # (T,54) float32
            # 专家动作
            actions = g["actions"][()]                               # (T,7) float64

            # 简单一致性检查
            T = actions.shape[0]
            for name, arr in {
                "front_rgb": front_rgb, "wrist_rgb": wrist_rgb, "ee_pose": ee_pose,
                "joint_forces": joint_forces, "contact_forces": contact_forces,
                "joint_states": joint_states
            }.items():
                if arr.shape[0] != T:
                    raise ValueError(f"{demo_key}:{name} 时间长度 {arr.shape[0]} 与 actions {T} 不一致")

            if not per_timestep:
                # 一个 demo → 一个 npz（轨迹级别，训练时用 --expand_trajectories）
                out_path = os.path.join(out_dir, f"{demo_key}.npz")
                np.savez_compressed(
                    out_path,
                    front_rgb=front_rgb.astype(np.uint8),
                    wrist_rgb=wrist_rgb.astype(np.uint8),
                    ee_pose=ee_pose.astype(np.float32),
                    joint_forces=joint_forces.astype(np.float32),
                    contact_forces=contact_forces.astype(np.float32),
                    joint_states=joint_states.astype(np.float32),
                    action=actions.astype(np.float32),
                )
                print(f"  Saved: {out_path} (T={T})")
            else:
                # 每个时间步 → 一个 npz（体量更大，但不需 --expand_trajectories）
                for t in range(T):
                    out_path = os.path.join(out_dir, f"{demo_key}_t{t:06d}.npz")
                    np.savez_compressed(
                        out_path,
                        front_rgb=front_rgb[t].astype(np.uint8),
                        wrist_rgb=wrist_rgb[t].astype(np.uint8),
                        ee_pose=ee_pose[t].astype(np.float32),
                        joint_forces=joint_forces[t].astype(np.float32),
                        contact_forces=contact_forces[t].astype(np.float32),
                        joint_states=joint_states[t].astype(np.float32),
                        action=actions[t].astype(np.float32),
                    )
                print(f"  Saved: {demo_key}_t*.npz x{T}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Convert a single HDF5 (with /data_frames/demo_*) to NPZ")
    ap.add_argument("--h5_dir", type=str, default="/home/wdy02/wdy_program/simulation_plus/IsaacLab/wdy_data/ONE_PEG_IN_HOLE/h5_file/ONE_PEG_IN_HOLE_final.h5", help="Path to directory containing .h5 files")
    ap.add_argument("--out_dir", type=str, default="/home/wdy02/wdy_program/simulation_plus/IsaacLab/wdy_data/bc_data/npz/ONE_PEG_IN_HOLE_test", help="Directory to save .npz files")
    ap.add_argument("--per_timestep", action="store_true", help="若给出此开关，则每个时间步保存一个 .npz")
    args = ap.parse_args()
    convert_one_h5(args.h5_dir, args.out_dir, per_timestep=args.per_timestep)
