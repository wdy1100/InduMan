# InduMan: A Benchmark for Long-Horizon, Contact-Rich Robotic Manipulation in Industrial

   A novel dataset, termed InduMan, is introduced for robotic manipulation in industrial environments, with an emphasis on long-horizon and contact-rich tasks. The dataset is built upon the NIST#1 assembly board and real industrial components,encompassing 26 assembly and disassembly tasks that involve fine alignment and complex force interactions.

## Prerequisites

* Ubuntu 20.04 or above
* Python 3.9
* Isaac Sim/Lab 4.5 or above
* gymnasium 1.2 or above

## Installation

Install Isaac Sim and Isaac Lab according to their websites

Isaac Sim: [https://github.com/isaac-sim/IsaacSim](https://github.com/isaac-sim/IsaacSim)

Isaac Lab: [https://github.com/isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab)

We adopt the binary installation method. Additionally, it is recommended to download the assets to your local device and remember to modify the path([Tips for assets installation](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/installation/install_faq.html)).

## Usage

Once you have installed Isaac Lab, see how to link Sim and Lab: [https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/binaries_installation.html](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/binaries_installation.html)

make a python environment `isaaclab.sh -c name of your env`

Normally, you can put this repo in your new project generated using [External project or Internal task](https://isaac-sim.github.io/IsaacLab/main/source/overview/own-project/template.html), or you can put it under Isaac Lab folder, which looks like IsaacLab(or your name of Isaac Lab)/InduMan

### Normal usage of scene:

Use code below to collect some json files for replay.

```python
python InduMan/run_main.py --task_name=(have a look at the task_config) --task_type=assembly/disassembly(you can distinguish by task name) --headless(True if you try to test your installation)
```

Replay the task you have done using json files.

InduMan/scripts/replay_logging_assembly.py This file only collect successful demos for assembly task.

InduMan/scripts/replay_logging_assembly_failed.py This file collect successful and failed demos for assembly task.

InduMan/scripts/replay_logging_disassembly.py This file only collect successful demos for disassembly task.

InduMan/scripts/replay_logging_disassembly.py This file only collect successful and failed demos for disassembly task.

### Try implicit Q-learning:

Before training a agent for implicit Q-learning, convert the h5 data collected by the environment into the required format.

You can check code in InduMan/implicit_q_learning/extract_feature.py

```python
python InduMan/implicit_q_learning/train_offline.py --env_name (choose one from task_config) --env_type (according to your task) --data_path (where your data put)
```

### Try behavioral cloning

Before training a agent for behavioral cloning, convert the h5 data collected by the environment into the required format.

You can check code in InduMan/utils/convert_h5_pkl.py

```python
python InduMan/robot_learning/run_BC.py --encoder_type=r3m --demo_path=(where does your pkl data put) --log_root_dir=(where do the results you want to put) --env (choose) --env_type (according to your env)
```

Before training a agent for implicit Q-learning, convert the h5 data collected by the environment into the required format.
