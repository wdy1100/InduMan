
"""
This script converts a custom HDF5 expert data file to a standard .pkl format for BC training.

Usage :
python convert_h5file.py --input=<path_to_h5_file> --output_dir=<path_to_output_dir> --single_file.
"""

import h5py
import numpy as np
import pickle
import argparse
import os
from tqdm import tqdm

def _has(obj, key):
    try:
        obj[key]
        return True
    except KeyError:
        return False
    
def convert_h5_to_pkl(h5_path, output_dir=None, save_per_demo=True):
    """
    Convert custom HDF5 expert data to standard .pkl format (for IL training)
    Args:
        h5_path (str): Path to the input .h5 
        fileoutput_dir (str): Output directory. If None, use the same directory as the h5 file
        save_per_demo (bool): Whether to save each demonstration as a separate .pkl file; 
        otherwise save as a single demos.pkl file
    """
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"H5 file not found: {h5_path}")

    if output_dir is None:
        base_name = os.path.splitext(os.path.basename(h5_path))[0]
        output_dir = os.path.join("converted_demos", base_name)
    os.makedirs(output_dir, exist_ok=True)

    with h5py.File(h5_path, 'r') as f:
        if not _has(f, "/data_frames"):
            raise KeyError("The HDF5 file lacks the '/data_frames' group")
        df = f["/data_frames"]

        demos = []
        demo_groups = [key for key in df.keys() if key.startswith('demo_')]
        demo_groups = sorted(demo_groups)  # sort by name

        print(f"Found {len(demo_groups)} demonstrations. Converting...")

        for group_name in tqdm(demo_groups, desc="Converting demos"):
            print(f"Converting demo: {group_name}")
            grp = df[group_name]

            # extract basic fields
            actions = grp['actions'][()]  # shape: (T, 7)
            rewards = grp['rewards'][()]    # shape: (T,)
            T = len(actions)

            # build the dones array where only the final step is True
            dones = np.zeros(T, dtype=bool)
            dones[-1] = True  # the final step, done = True

            # construct the obs dictionary (fields can be added or removed as needed)
            obs_components = {
                # continuous state vector
                'ee_poses': grp['observations/ee_poses'][()],           # (T, 7)
                'gripper_joint_states': grp['gripper_joint_states'][()], # (T, 2)
                'robot_joint_states': grp['robot_joint_states'][()],     # (T, 14)
                'robot_measured_joint_forces': grp['robot_measured_joint_forces'][()],  # (T, 54)
                # discrete state vector
                'dual_hole_poses': grp['dual_hole_poses'][()],           # (T, 7)
                'dual_peg_poses': grp['dual_peg_poses'][()],             # (T, 7)
                'dual_hole_contact_forces': grp['dual_hole_contact_forces'][()],  # (T, 3)
                # image observation (optional, note memory usage)
                'agentview_rgb': grp['observations/agentview_rgb'][()],   # (T, 128, 128, 3)
                'hand_camera_rgb': grp['observations/hand_camera_rgb'][()], # (T, 128, 128, 3)
            }

            # concatenate all state observations into a single array
            parts_state_obs_list = [
                obs_components['dual_hole_poses'],
                obs_components['dual_peg_poses'],
                obs_components['dual_hole_contact_forces']
            ]
            parts_state_obs = np.concatenate(parts_state_obs_list, axis=-1)  # (T, 29)

            robot_state_obs = np.concatenate([
                obs_components['ee_poses'],
                obs_components['gripper_joint_states'],
                obs_components['robot_joint_states'],
                obs_components['robot_measured_joint_forces']
            ], axis=-1)  # (T, 77)

            obs = []
            for i in range(len(parts_state_obs)):
                ob = {
                    "parts_state": parts_state_obs[i],
                    "robot_state": robot_state_obs[i],
                    "agentview_img": obs_components['agentview_rgb'][i],
                    "hand_camera_img": obs_components['hand_camera_rgb'][i],
                }
                obs.append(ob)


            demo = {
                "obs": obs,           # observations in the form of a dictionary
                "actions": actions,
                "rewards": rewards,
                "dones": dones,
                # extensible to other fields
            }

            demos.append(demo)

            # if selected, save each demo as a separate file
            if save_per_demo:
                demo_pkl_path = os.path.join(output_dir, f"{group_name}.pkl")
                with open(demo_pkl_path, 'wb') as pf:
                    pickle.dump(demo, pf)
                print(f"Saved {demo_pkl_path}")

        # if not selected, save all demos to one file
        if not save_per_demo:
            all_demos_pkl = os.path.join(output_dir, "demos.pkl")
            with open(all_demos_pkl, 'wb') as pf:
                pickle.dump(demos, pf)
            print(f"All demos saved to {all_demos_pkl}")

    print(f"✅ Conversion completed! Output saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert H5 expert data to PKL format.")
    parser.add_argument("--input", type=str, required=True, help="Path to input .h5 file")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--single_file", action='store_true',
                        help="Save all demos into one demos.pkl instead of separate files")

    args = parser.parse_args()

    convert_h5_to_pkl(
        h5_path=args.input,
        output_dir=args.output_dir,
        save_per_demo=not args.single_file
    )