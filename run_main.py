"""
    This is the main file to run the Franka Gym environment.
    Specify the task name and task type in the command line arguments,
    choose a device to control the robot (keyboard/joystick), and select
    where to load the objects assets (isaac sim assets path/program folder).
    The default values are set to assemble task and joystick device, and use
    isaac sim assets path.
    
"""

import sys
import threading
import queue
import os
import yaml

from isaacsim import SimulationApp


sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
# actual_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
# print(actual_parent_dir)
from InduMan.utils import get_assets_path, get_task_cfg_path
from InduMan.utils import KeyboardController, JoystickController

def parser_args():
    import argparse
    parser = argparse.ArgumentParser(description="Franka Gym Environment")
    parser.add_argument('--task_name', type=str, default='RECTANGLE_PEG_OUT_HOLE', help='Name of the task to run')
    parser.add_argument('--task_type',type=str, default='disassembly', help='Type of the environment to run')
    parser.add_argument('--task_assets_path', type=str, default=get_assets_path(), help='Path to the task assets')
    parser.add_argument('--control_frequency',type=float, default=10.0, help='Control frequency in Hz')
    parser.add_argument('--device', type=str, default="keyboard", help='Device to control the robot (joystick/keyboard)')
    parser.add_argument('--headless', type=bool, default=False, help='Run simulation in headless mode')
    args = parser.parse_args()

    # decide which device to use, default is keyboard, convenient for testing
    if args.device is None:
        # use ANSI color codes to display colored prompts
        print("\033[94mChoose a device to control the robot：\033[0m")         # blue
        print("\033[93m1. Joystick\033[0m")      # yellow
        print("\033[92m2. Keyboard\033[0m")      # green
        choice = input("\033[96mEnter a number (1/2): \033[0m").strip()  # light blue input prompt
        if choice == '1':
            args.device = 'joystick'
        elif choice == '2':
            args.device = 'keyboard'
        else:
            print("Invalid input, using keyboard")
            args.device = 'keyboard'
    # # decide where to load the objects assets, default is isaac sim assets path
    ## convenient for testing, you can annotate this if you do not need.
    print("\033[94mSelect where to load the objects assets：\033[0m")         # blue
    print("\033[93m1. Use the assets in the program folder\033[0m")      # yellow
    print("\033[92m2. Use isaac sim assets\033[0m")      # green
    choice = input("\033[96mChoose a number (1/2): \033[0m").strip()  # light blue input prompt
    if choice == '1':
        args.task_assets_path = get_assets_path()
    elif choice == '2':
        args.task_assets_path = None
    else:
        print("Invalid input, using isaac sim assets path")
        args.task_assets_path = None
    
    if args.task_type is None:
        # decide which task environment to run, default is assemble
        print("\033[94mSelect which task environment to run:\033[0m")         # blue
        print("\033[93m1. Assemble\033[0m")      # yellow
        print("\033[92m2. Disassemble\033[0m")      # green
        choice = input("\033[96mChoose a number (1/2): \033[0m").strip()  # light blue input prompt
        if choice == '1':
            args.task_type = 'assembly'
        elif choice == '2':
            args.task_type = 'disassembly'
        else:
            print("Invalid input, using assemble")
            args.task_type = 'assemble'
    return args

def main(args=None):
    simulation_app = SimulationApp({"headless": args.headless})
    # load task settings
    task_name = args.task_name
    task_config_path = os.path.join(get_task_cfg_path(), f"{task_name}.yaml")
    with open(task_config_path, 'r') as f:
        task_config = yaml.safe_load(f)

    if args.task_type == 'disassembly':
        from InduMan.scripts import FrankaDisassemblyGym
        my_env = FrankaDisassemblyGym(is_replay=False,
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
    elif args.task_type == 'assembly':
        from InduMan.scripts import FrankaAssemblyGym
        my_env = FrankaAssemblyGym(is_replay=False,
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
        raise ValueError(f"Unsupported task type: {args.task_type}")

    running_flag = [True]  # use list to make it mutable in threads
    action_queue = queue.Queue(maxsize=1)
    # shared data between main thread and device controller thread
    shared_data = {
        'x_button': False,
        'y_button': False,
        'not_saved': False,  # 用于标记是否需要保存
        'gripper_close': False,  # 夹爪状态
    }

    # create thread lock
    shared_lock = threading.Lock()
    if args.device == 'keyboard':
        device = KeyboardController(shared_data=shared_data, 
                                               lock=shared_lock, 
                                               action_queue=action_queue,
                                                 running_flag=running_flag,
                                                 control_frequency=args.control_frequency,
                                                 )
    elif args.device == 'joystick':
        device = JoystickController(shared_data=shared_data, 
                                               lock=shared_lock, 
                                               action_queue=action_queue,
                                                 running_flag=running_flag,
                                                 control_frequency=args.control_frequency,
                                                 )
    else:
        raise ValueError(f"Unsupported device: {args.device}")

    while simulation_app.is_running():
        my_env.world.step(render=True)
        try :
            latest_action  = action_queue.get_nowait()
        except queue.Empty:
            latest_action = None
        with shared_lock:
            if shared_data['x_button']:
                print("X button pressed, Resetting the world...")
                shared_data['gripper_close'] = False
                my_env.reset()

            if shared_data['y_button']:
                print("Y button pressed, saving demonstration...")
                my_env.logging_save()
                shared_data['gripper_close'] = False
                my_env.reset()

            if shared_data['not_saved']:
                print("Not saved, exiting...")
                break

        if my_env.world.is_playing():
            if latest_action is not None:
                # apply action
                _, _, _, _, _ = my_env.step(action=latest_action)

            if my_env.is_success:
                    for i in range(10):  # render a few more frames
                        my_env.world.step(render=True)
                    print("Episode finished!")
                    my_env.logging_save()
                    shared_data['gripper_close'] = False
                    my_env.reset()

    device.stop()
    simulation_app.close()

if __name__ == '__main__':
    args = parser_args()
    main(args)