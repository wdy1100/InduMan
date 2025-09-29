
"""
General script: Remove the demo group in the HDF5 file where the demo_name is on the blacklist

"""
import argparse
import os
import shutil
import time
import h5py

def load_bad_names(args):
    bad = []
    if args.bad_list:
        for s in args.bad_list.split(','):
            ss = s.strip()
            if ss: bad.append(ss)
    # duplicate removal
    return list(dict.fromkeys(bad))

def find_groups_with_demo_names(h5path, bad_names):
    to_delete = []
    with h5py.File(h5path, 'r') as f:
        def visitor(name, obj):
            # only check group
            if isinstance(obj, h5py.Group):
                if 'demo_name' in obj.attrs:
                    demo = obj.attrs['demo_name']
                    # handle bytes situation
                    if isinstance(demo, (bytes, bytearray)):
                        try:
                            demo = demo.decode('utf-8')
                        except Exception:
                            demo = demo.decode(errors='ignore')
                    # sometimes attr is numpy.bytes_ or numpy.str_
                    demo = str(demo)
                    if demo in bad_names:
                        to_delete.append(name)
        f.visititems(visitor)
    return to_delete

def backup_file(h5path):
    dirn = os.path.dirname(h5path)
    base = os.path.basename(h5path)
    ts = time.strftime("%Y%m%d_%H%M%S")
    bak = os.path.join(dirn, f"{base}.bak.{ts}")
    shutil.copy2(h5path, bak)
    return bak

def delete_groups(h5path, group_paths):
    with h5py.File(h5path, 'r+') as f:
        for g in group_paths:
            if g in f:
                del f[g]
            else:
                # It might be a sub-path, such as 'demo_group/sub'
                try:
                    del f[g]
                except Exception as e:
                    print(f"Failed to delete {g}: {e}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5", default="/home/wdy02/wdy_program/simulation_plus/IsaacLab/wdy_data/GEARS_IN_PEG/h5_file/GEARS_IN_PEG.h5", help="HDF5 file path")
    p.add_argument("--bad_list", default=["1757206905.558829_episode_602.json","1757207111.614786_episode_542.json","1757209491.7690885_episode_721.json","1757209163.2755766_episode_304.json"],help="逗号分隔的不合理 demo 名列表")
    p.add_argument("--backup", action="store_true", help="Make a file backup before making any modifications")
    p.add_argument("--dry_run", action="store_true", help="Only list the groups that will be deleted and do not delete them")
    p.add_argument("--yes", default=True, help="Automatic confirmation deletion (skip interactive prompts)")
    args = p.parse_args()

    if not os.path.isfile(args.h5):
        print("HDF5 file does not exist:", args.h5)
        return

    bad_names = load_bad_names(args)
    if not bad_names:
        print("No invalid demo names provided (--bad-list)")
        return

    matches = find_groups_with_demo_names(args.h5, bad_names)
    if not matches:
        print("No matching demo was found (no need to delete)")
        return

    print(f"In the file {args.h5}, find the {len(matches)} groups that will be deleted:")
    for m in matches:
        print("  ", m)

    if args.dry_run:
        print("dry-run mode, no changes will be made")
        return

    if args.backup:
        bak = backup_file(args.h5)
        print("File backup created:", bak)

    if not args.yes:
        ans = input("Are you sure to delete the above groups? (y/n): ")
        if ans.lower() != 'y':
            print("Deletion cancelled.")
            return

    print("Start deleting ...")
    delete_groups(args.h5, matches)
    print("Deletion complete. Please check the file again.")


if __name__ == "__main__":
    main()