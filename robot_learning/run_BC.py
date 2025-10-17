import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
# actual_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
# print(actual_parent_dir)
"""
run the BC experiment:
python wdy_rolf/run_BC.py --demo_path=/home/wdy02/wdy_program/simulation_plus/IsaacLab/wdy_data/bc_data/pkl/demos.pkl --log_root_dir=/home/wdy02/wdy_program/simulation_plus/IsaacLab/wdy_data/bc_data/bc_ckpts

"""

from main import run

if __name__ == "__main__":
    # torch.multiprocessing.set_start_method('spawn')

    run()
