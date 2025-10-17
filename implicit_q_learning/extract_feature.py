import pickle
from pathlib import Path
import numpy as np
import torch
from absl import app, flags

"""
 code to extract features from demonstrations

 not use r3m :python wdy_file/implicit_q_learning/extract_feature.py --demo_dir=/home/wdy02/wdy_program/simulation_plus/IsaacLab/wdy_data/bc_data/pkl/demos.pkl --out_file_path=/home/wdy02/wdy_program/simulation_plus/IsaacLab/wdy_data/iql_data/pkl/DUAL_HOLE_IN_PEG.pkl
 use r3m :python wdy_file/implicit_q_learning/extract_feature.py --demo_dir=/home/wdy02/wdy_program/simulation_plus/IsaacLab/wdy_data/bc_data/pkl/demos.pkl --out_file_path=/home/wdy02/wdy_program/simulation_plus/IsaacLab/wdy_data/iql_data/pkl/DUAL_HOLE_IN_PEG.pkl --use_r3m=True
"""

FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", "DUAL_HOLE_IN_PEG", "Furniture name.")
flags.DEFINE_string("demo_dir", "square_table_parts_state", "Demonstration dir.")
flags.DEFINE_string("out_file_path", None, "Path to save converted data.")
flags.DEFINE_boolean("use_r3m", True, "Use r3m to encode images.")
flags.DEFINE_boolean("use_vip", False, "Use vip to encode images.")
flags.DEFINE_integer("num_threads", int(8), "Set number of threads of PyTorch")
flags.DEFINE_integer("num_demos", None, "Number of demos to convert")
flags.DEFINE_integer('batch_size', 256, 'Batch size for encoding images')

def main(_):
    if FLAGS.num_threads > 0:
        print(f"Setting torch.num_threads to {FLAGS.num_threads}")
        torch.set_num_threads(FLAGS.num_threads)

    env_name = FLAGS.env_name
    demo_file_path = Path(FLAGS.demo_dir)  # the path to a single .pkl file

    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []

    # === load the image encoder（R3M or VIP）===
    if FLAGS.use_r3m:
        from r3m import load_r3m
        encoder = load_r3m('resnet50')
        # feature_dim = 2048
        print("Using R3M (ResNet50) for encoding images.")
    elif FLAGS.use_vip:
        from vip import load_vip
        encoder = load_vip()
        # feature_dim = 1024
        print("Using VIP for encoding images.")
    else:
        encoder = None
        # feature_dim = None

    if encoder is not None:
        encoder.eval()
        for param in encoder.parameters():
            param.requires_grad = False
        encoder.to('cuda')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # === read a single .pkl file of multiple demonstrations ===
    print(f"Loading demonstrations from {demo_file_path}...")
    with open(demo_file_path, 'rb') as f:
        trajectories = pickle.load(f)  # list of demos

    if not isinstance(trajectories, (list, tuple)):
        raise ValueError(f"Expected a list of demos, but got {type(trajectories)}")

    num_loaded = len(trajectories)
    num_to_process = FLAGS.num_demos if FLAGS.num_demos else num_loaded
    print(f"Found {num_loaded} demos, processing {num_to_process} demos.")

    # === process each trajectory ===
    for i, demo in enumerate(trajectories[:num_to_process]):
        if i >= num_to_process:
            break

        print(f"Processing demo [{i+1}/{num_to_process}] with {len(demo['actions'])} steps...")

        obs_list = demo["obs"]
        actions = demo["actions"]
        rewards = demo.get("rewards", [0] * len(actions))
        dones = demo.get("dones", [False] * len(actions))

        T = len(obs_list)

        if T == 0:
            print(f"Warning: Demo {i} has no observations, skipping.")
            continue

        # check the length consistency
        if len(actions) != T - 1 and len(actions) != T:
            print(f"Warning: Action length {len(actions)} mismatch with obs length {T}, skipping.")
            continue

        if encoder is not None:
            # define a function to convert image to tensor (support HWC/CHW auto-recognition)
            def to_tensor(img):
                if isinstance(img, np.ndarray):
                    if img.ndim != 3:
                        raise ValueError(f"Image should be 3D, got {img.ndim}D")
                    if img.shape[-1] == 3:  # HWC
                        img = np.transpose(img, (2, 0, 1))  # → CHW
                    elif img.shape[0] == 3:  # CHW
                        pass
                    else:
                        raise ValueError(f"Unrecognized image format with shape: {img.shape}")
                    return torch.from_numpy(img)
                else:
                    raise TypeError(f"Expected np.ndarray, got {type(img)}")

            try:
                img1_batch = torch.stack([to_tensor(ob["agentview_img"]) for ob in obs_list])
                img2_batch = torch.stack([to_tensor(ob["hand_camera_img"]) for ob in obs_list])
            except Exception as e:
                print(f"Error processing images in demo {i}: {e}")
                continue

            img1_batch = img1_batch.float().to(device) / 255.0  # (T, 3, H, W)
            img2_batch = img2_batch.float().to(device) / 255.0

            # encode in batches
            img1_features = []
            img2_features = []

            with torch.no_grad():
                for start_idx in range(0, T, FLAGS.batch_size):
                    end_idx = min(start_idx + FLAGS.batch_size, T)
                    feat1 = encoder(img1_batch[start_idx:end_idx]).cpu().numpy()
                    feat2 = encoder(img2_batch[start_idx:end_idx]).cpu().numpy()
                    img1_features.append(feat1)
                    img2_features.append(feat2)

            img1_encoded = np.concatenate(img1_features, axis=0).astype(np.float32)
            img2_encoded = np.concatenate(img2_features, axis=0).astype(np.float32)
        else:
            img1_encoded = None
            img2_encoded = None

        # build transitions
        for t in range(T - 1):
            # current state
            if encoder is not None:
                image1 = img1_encoded[t]
                image2 = img2_encoded[t]
                next_image1 = img1_encoded[t + 1]
                next_image2 = img2_encoded[t + 1]
            else:
                # original image (assuming CHW -> convert to HWC for better storage)
                def chw_to_hwc(img):
                    if img.ndim == 3 and img.shape[0] == 3:
                        return np.transpose(img, (1, 2, 0))
                    return img

                image1 = chw_to_hwc(obs_list[t]["agentview_img"])
                image2 = chw_to_hwc(obs_list[t]["hand_camera_img"])
                next_image1 = chw_to_hwc(obs_list[t + 1]["agentview_img"])
                next_image2 = chw_to_hwc(obs_list[t + 1]["hand_camera_img"])

            # construct the observation dictionary
            if encoder is not None:
                obs_.append({
                    'image1': image1,
                    'image2': image2,
                    'robot_state': obs_list[t]["robot_state"].astype(np.float32),
                    # 'parts_state': obs_list[t]["parts_state"].astype(np.float32),
                })

                next_obs_.append({
                    'image1': next_image1,
                    'image2': next_image2,
                    'robot_state': obs_list[t + 1]["robot_state"].astype(np.float32),
                    # 'parts_state': obs_list[t + 1]["parts_state"].astype(np.float32),
                })
            else:
                obs_.append(obs_list[t]["robot_state"].astype(np.float32))
                next_obs_.append(obs_list[t + 1]["robot_state"].astype(np.float32))

            action_.append(actions[t])
            reward_.append(rewards[t])
            done_.append(1 if dones[t] else 0)  # use the original dones, with a fallback to checking if it's the final step

    # === build the final dataset ===
    # not use encoder:
    if encoder is None:
        dataset = {
            "observations": np.array(obs_),
            "next_observations": np.array(next_obs_),
            "actions": np.array(action_),
            "rewards": np.array(reward_),
            "terminals": np.array(done_),  # check and tag episode end status
        }
    # use encoder:
    else:
        dataset = {
            "observations": obs_,
            "next_observations": next_obs_,
            "actions": np.array(action_),
            "rewards": np.array(reward_),
            "terminals": np.array(done_),  # check and tag episode end status
        }

    # === save the results ===
    out_path = FLAGS.out_file_path or f"data/Image/{env_name}.pkl"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("wb") as f:
        pickle.dump(dataset, f)
        print(f"✅ Converted dataset saved to {out_path}")
        print(f"   Total transitions: {len(action_)}")
        print(f"   Number of demos processed: {min(num_to_process, num_loaded)}")

if __name__ == "__main__":
    app.run(main)
