"""
    This file is the gym environment for assemble tasks.
    It will load the task and the robot, and provide the interface for the agent to interact with the environment.
"""

import numpy as np
import gymnasium as gym
import sys
import os
import time
import carb

from isaacsim.core.api import World
from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.robot.manipulators.examples.franka import KinematicsSolver
from isaacsim.core.utils.stage import add_reference_to_stage, get_stage_units
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.core.utils.rotations import matrix_to_euler_angles, euler_angles_to_quat, rot_matrix_to_quat
from isaacsim.core.api.objects import VisualSphere, VisualCylinder,VisualCuboid
from isaacsim.core.prims import GeometryPrim,RigidPrim,XFormPrim
from isaacsim.core.api.materials import OmniPBR
from omni.isaac.sensor import Camera
from pxr import UsdLux, Gf

from .assembly_task import AssemblyTask
from InduMan.utils import get_default_data_path, get_assets_path, ForceMonitor

class FrankaAssemblyGym(gym.Env):
    def __init__(self,
                 save_path=None,
                 render=True,
                 is_replay=False,
                 stage_units = 1.0,
                 image_size = (128,128),
                 physics_dt = 60.0,
                 rendering_dt = 60.0,
                 max_episode_steps = 1000,
                 table_height = 0.0,
                 task_name = "TASK_NAME",
                 task_message = ["TASK_MESSAGE"],
                 objects_to_manipulate = ["OBJECT_NAME"],
                 objects_to_interact = ["OBJECT_NAME"],
                 support = None,
                 task_assets_path = None,
                 gripper_opened_position = np.array([0.05, 0.05]),
                 ):
        self.save_path = save_path if save_path is not None else get_default_data_path()
        self.is_replay = is_replay
        self.stage_units = stage_units
        self.image_size = image_size
        self.physics_dt = physics_dt
        self.rendering_dt = rendering_dt
        self.max_episode_steps = max_episode_steps

        self.table_height = table_height
        self.task_name = task_name
        self.task_message = task_message
        self.objects_to_manipulate = objects_to_manipulate
        self.objects_to_interact = objects_to_interact 

        self.episode_steps = 0
        self.is_success = False

        #  a class for contact force monitor
        self.force_monitor = ForceMonitor(threshold=5, min_interval=1)

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(7,))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(77,))

        self.assets_root_path = get_assets_root_path()
        if self.assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
            self.simulation_app.close()
            sys.exit()  

        self.world = World(stage_units_in_meters=self.stage_units)
        self.world.set_simulation_dt(physics_dt = 1./ self.physics_dt, 
                                     rendering_dt = 1. / self.rendering_dt)
        # add data logger
        self.data_logger = self.world.get_data_logger()
        self.data_logger.add_data_frame_logging_func(self.human_control_logging)
        self.data_path = os.path.join(self.save_path, f'{task_name}')
        
        self.world.scene.add_default_ground_plane()
        
        # add task
        task_assets_path = self.assets_root_path if task_assets_path is None else task_assets_path
        self.task = AssemblyTask(name = self.task_name,
                            task_message = self.task_message,
                            is_replay = self.is_replay,
                            table_height=self.table_height,
                            assets_root_path=task_assets_path,
                            objects_to_manipulate_dict = self.objects_to_manipulate,
                            objects_to_interact_dict = self.objects_to_interact,
                            support = support,
                            )
        self.world.add_task(self.task)

        set_camera_view(
                eye=[1.3, 0.,1.6], target=[0.00, 0.00, 0.8], 
                camera_prim_path="/OmniverseKit_Persp"
                )  # set camera view
        
        self.add_base_env(gripper_opened_position=np.array(gripper_opened_position))
        self.add_walls()
        self.add_shpere_light()

        # get franka metadata for joint names and indices
        franka_metadata = self.franka._articulation_view._metadata 
        self.forces_joint_indices = [1 + franka_metadata.joint_indices[name] for name in self.franka.dof_names]

        # get objects in task it's a dict with names and properties
        self.objects_of_task = self.task.get_task_objects()
        self.world.reset()

    def reset(self):
        self.episode_steps = 0
        self.is_success = False
        self.data_logger.reset()
        self.world.reset()
        self.task.reset_objects_position()
        init_obs = self.get_observation()  # get initial observation
        return init_obs

    def add_base_env(self,
                    table_position=np.array([[0.8,0.2,1.03]]),
                   stand_position=np.array([[0.0,0.0,0.775]]),
                   robot_init_position = np.array([0.09013208, -0.526969 ,  -0.09634338, -2.5128212,  -0.04378112,  1.9551778,0.8178173]),
                   gripper_opened_position= np.array([0.05, 0.05])
                   ):
        # add a stand for robot
        stand_path = self.assets_root_path + "/Isaac/Props/Mounts/Stand/stand.usd"
        table_path = self.assets_root_path + "/Isaac/Props/Mounts/textured_table.usd"
        add_reference_to_stage(usd_path=table_path, prim_path="/World/table")
        add_reference_to_stage(usd_path=stand_path, prim_path="/World/stand")
        self.table = GeometryPrim(prim_paths_expr="/World/table",
                                  name="table_0",
                            positions=table_position / self.stage_units,
                            )
        self.stand = GeometryPrim(prim_paths_expr="/World/stand",
                                  name="stand_0",
                          positions=stand_position/ self.stage_units,
                          scales=np.array([[1.5,1.5,1.5]]),
                          )
        # add agentview camera
        self.camera_world = Camera(prim_path="/World/table/CameraWorld",
                        name="camera_1",
                        resolution=self.image_size,
                        frequency=self.rendering_dt,
                        orientation=np.array([0.,0.,0.,-1.]),
                        position=np.array([1,0.0,1.2]),
                        )
        self.camera_world.set_clipping_range(0.005, 10.0)
        self.camera_world.set_focal_length(1)
        
        # add a camera in right 
        right_ori = euler_angles_to_quat(np.array([np.pi/15,0,-np.pi/2]),extrinsic=False)
        self.camera_world_right = Camera(prim_path="/World/table/CameraWorld_right",
                        name="camera_2",
                        resolution=self.image_size,
                        frequency=self.rendering_dt,
                        orientation=right_ori,
                        translation=np.array([-0.05,0.25,0.2]),
                        )
        self.camera_world_right.set_clipping_range(0.005, 10.0)
        self.camera_world_right.set_focal_length(1)
        
        # add a camera in left
        left_ori = euler_angles_to_quat(np.array([-np.pi/15,0,np.pi/2]),extrinsic=False)
        self.camera_world_left = Camera(prim_path="/World/table/CameraWorld_left",
                        name="camera_3",
                        resolution=self.image_size,
                        frequency=self.rendering_dt,
                        orientation=left_ori,
                        translation=np.array([-0.05,-0.8,0.2]),
                        )
        self.camera_world_left.set_clipping_range(0.005, 10.0)
        self.camera_world_left.set_focal_length(1)


         # add a franka robot
        self.franka = Franka(prim_path="/World/stand/franka",
                             name="franka_0",
                             gripper_open_position=gripper_opened_position/get_stage_units(),
                             gripper_closed_position=np.array([0.0, 0.0]),
                             deltas=gripper_opened_position/get_stage_units(),
                             )
        # gripper class
        self.gripper = self.franka.gripper
        self.gripper.set_default_state(gripper_opened_position/get_stage_units())

        franka_position= np.concatenate([robot_init_position,self.gripper.joint_opened_positions])
        self.franka.set_joints_default_state(positions = franka_position)

        # add camera sensor(camera in hand)
        self.camera_hand = Camera(prim_path="/World/stand/franka/panda_hand/CameraHand_front",
                             name="camera_hand_front",
                             resolution=self.image_size,
                             frequency=self.rendering_dt,
                             orientation=np.array([-0.6428,0.,0.776,0.]),
                            translation=np.array([0.05,0.0,0.005]),
                             )
        self.camera_hand.set_clipping_range(0.005, 10.0)
        self.camera_hand.set_focal_length(1)
        # add a camera in hand back
        self.camera_hand_back = Camera(prim_path="/World/stand/franka/panda_hand/CameraHand_back",
                        name="camera_hand_back",
                        resolution=self.image_size,
                        frequency=self.rendering_dt,
                        orientation=np.array([-0.766044,0.,0.642788,0.]),
                    translation=np.array([-0.05,0.0,0.005]),
                        )
        self.camera_hand_back.set_clipping_range(0.005, 10.0)
        self.camera_hand_back.set_focal_length(1)
        # add a camera in hand center
        self.camera_hand_center = Camera(prim_path="/World/stand/franka/panda_hand/CameraHandCenter",
                        name="camera_hand_center",
                        resolution=self.image_size,
                        frequency=self.rendering_dt,
                        orientation=np.array([0.707107,0,-0.707107,0.]),
                    translation=np.array([0.0,0.003,0.07]),
                        )
        self.camera_hand_center.set_clipping_range(0.005, 10.0)
        self.camera_hand_center.set_focal_length(1)
        # add visual marker for hand
        self.tool_center_marker = VisualSphere(prim_path="/World/stand/franka/panda_hand/tool_center/Sphere",
                                          name="tool_center_marker",
                                      radius=0.001,color=np.array([0.0,1.0,0.0]))
        self.hand_marker = VisualCylinder(prim_path="/World/stand/franka/panda_hand/tool_center/Cylinder",
                                     name="hand_marker",
                                    translation=np.array([0,0,0.1]),
                                    radius=0.0005,height=0.3, color=np.array([1,0,0]))
        
        self.world.scene.add(self.table)
        self.world.scene.add(self.stand)
        self.world.scene.add(self.camera_world)
        self.world.scene.add(self.camera_world_right)
        self.world.scene.add(self.camera_world_left)

        self.world.scene.add(self.franka)
        # self.world.scene.add(tool_center_marker)
        # self.world.scene.add(hand_marker)
        self.world.scene.add(self.camera_hand)
        self.world.scene.add(self.camera_hand_back)
        self.world.scene.add(self.camera_hand_center)
        self.world.reset()

        self.controller = KinematicsSolver(self.franka)
        self.controller._kinematics_solver.set_default_position_tolerance(tolerance=0.00001)
        self.controller._kinematics_solver.set_default_orientation_tolerance(tolerance=0.004)
        self.articulation_controller = self.franka.get_articulation_controller()

    def add_shpere_light(self,radius=0.25,color=np.array([1.0,1.0,1.0]),
                    intensity=50000.0,position=np.array([1.2,0.0,1.5])):
        # add a sphere light
        self.stage = self.world.stage
        light_prim = UsdLux.SphereLight.Define(self.stage, "/World/SphereLight")
        light_prim.CreateRadiusAttr(radius)
        light_prim.CreateColorAttr(Gf.Vec3f(color[0],color[1],color[2]))
        light_prim.CreateIntensityAttr(intensity)
        light_prim.AddTranslateOp().Set(Gf.Vec3f(position[0],position[1],position[2]))

    def add_walls(self,):
        # add walls
        back_wall = VisualCuboid(prim_path="/World/Back_wall",name="wall_0",
                                 position=np.array([-2.,0.,1.5]),
                                 scale=np.array([0.001,4.0,3]),
                                #  visual_material=
                            )
        right_wall = VisualCuboid(prim_path="/World/Right_wall",name="wall_1",
                                 position=np.array([0.,2.,1.5]),
                                 scale=np.array([4.0,0.001,3]),
                                #  visual_material=
                            )
        left_wall = VisualCuboid(prim_path="/World/Left_wall",name="wall_2",
                                 position=np.array([0.,-2.,1.5]),
                                 scale=np.array([4.0,0.001,3]),
                                #  visual_material=
                            )
        front_wall = VisualCuboid(prim_path="/World/Front_wall",name="wall_3",  
                                 position=np.array([2.,0.,1.5]),
                                 scale=np.array([0.001,4.0,3.0]),
                                #  visual_material=
                            )
        texture_path = os.path.join(get_assets_path(),'textures/wall_texture.png')
        wall_material = OmniPBR(prim_path="/World/wall_material",
                                name="wall_material",
                                texture_path=texture_path,)
        back_wall.apply_visual_material(wall_material)
        right_wall.apply_visual_material(wall_material)
        left_wall.apply_visual_material(wall_material)
        front_wall.apply_visual_material(wall_material)
        self.world.scene.add(back_wall)
        self.world.scene.add(right_wall)
        self.world.scene.add(left_wall)
        self.world.scene.add(front_wall)
        self.world.reset()     

    def step(self, action):
        print(f"execute action: {action}")
        self.control_action = action
        ee_position,ee_ori_mat = self.controller.compute_end_effector_pose()
        ee_euler = matrix_to_euler_angles(mat=ee_ori_mat,extrinsic=False)
        translation = ee_position + self.control_action[:3]
        rotation = euler_angles_to_quat(ee_euler + self.control_action[3:6],extrinsic=False)
        joint_actions,success = self.controller.compute_inverse_kinematics(
            target_position=translation,
            target_orientation=rotation)
        if success:
            self.articulation_controller.apply_action(joint_actions)
            #gripper action
            if self.control_action[6] == 1:
                self.gripper.close()
            else:
                # make sure gripper only open once
                if np.all(self.gripper.get_joint_positions() >= self.gripper.joint_opened_positions):
                    pass
                elif np.all(self.gripper.get_joint_positions() < self.gripper.joint_opened_positions- 0.001):
                    self.gripper.open()
        else:
            print("IK failed")
        self.observation = self.get_observation()
        done = self.is_done()
        reward = 1.0 if self.is_success else 0.0
        info = self.get_info()
        self.episode_steps += 1
        if not self.is_replay:
            self.is_first_action()
    
        return self.observation, reward, done,self.is_success, info
    
    def is_done(self):
        is_done = False
        if self.episode_steps >= self.max_episode_steps:
            is_done = True
        if self.task.is_done():
            is_done = True
            self.is_success = True
        return is_done

    def is_first_action(self):
        # if first action , begin to data logging
        if self.episode_steps == 1:
            if self.task.support['need']: # support object depends the manipulation object position
                joints_state = self.franka.get_joints_state()
                data = {
                        "episode_steps": self.episode_steps,
                        "actions": self.control_action.tolist(),
                        "robot_states": {
                            'joint_positions': joints_state.positions[:7].tolist(),
                        },
                        "objects": {},
                        "support": {},
                    }
                for name, obj in self.objects_of_task.items():
                    data["objects"][name] = {
                        "position": [round(float(x), 5) for x in np.round(obj.get_world_poses()[0], 5).flatten()],
                        "orientation": [round(float(x), 5) for x in np.round(obj.get_world_poses()[1], 5).flatten()]
                    }
                for name, obj in self.task.objects_support.items():
                    data["support"][name] = {
                        "position": [round(float(x), 5) for x in np.round(obj.get_world_pose()[0], 5).flatten()],
                        "orientation": [round(float(x), 5) for x in np.round(obj.get_world_pose()[1], 5).flatten()]
                    }
            else:
                joints_state = self.franka.get_joints_state()
                data = {
                        "episode_steps": self.episode_steps,
                        "actions": self.control_action.tolist(),
                        "robot_states": {
                            'joint_positions': joints_state.positions[:7].tolist(),
                        },
                        "objects": {},
                    }
                for name, obj in self.objects_of_task.items():
                    data["objects"][name] = {
                        "position": [round(float(x), 5) for x in np.round(obj.get_world_poses()[0], 5).flatten()],
                        "orientation": [round(float(x), 5) for x in np.round(obj.get_world_poses()[1], 5).flatten()]
                    }
            self.data_logger.add_data(data=data,
                                      current_time_step=self.world.current_time_step_index,
                                      current_time=self.world.current_time)
            self.data_logger.start()

    def get_observation(self):
        # get observation from the world
        joints_state = self.franka.get_joints_state()
        joint_force_torque = self.franka.get_measured_joint_forces(self.forces_joint_indices).flatten()
        

        hand_rgb = self.get_camera_observation(self.camera_hand,depth=False)
        agent_rgb = self.get_camera_observation(self.camera_world,depth=False)
        hand_back_rgb = self.get_camera_observation(self.camera_hand_back,depth=False)

        agent_left_rgb = self.get_camera_observation(self.camera_world_left,depth=False)
        agent_right_rgb = self.get_camera_observation(self.camera_world_right,depth=False)

        # get task observation
        task_obs = self.task.get_observations(with_forces=True)
        self.force_monitor.check_observation(task_obs)

        # for name, obj in self.objects_of_task.items():
        #     print(name, obj.get_world_poses()[0])
        ee_position,ee_ori_mat = self.controller.compute_end_effector_pose()
        ee_ori_quat = rot_matrix_to_quat(ee_ori_mat)
        ee_pose = np.concatenate([ee_position,ee_ori_quat])
        task_obs.update({
            "ee_pose": ee_pose,
            "joint_positions": joints_state.positions[:7],
            "joint_velocities": joints_state.velocities[:7],
            "gripper_state": self.gripper.get_joint_positions(),
            "joint_forces_torques": joint_force_torque,
            "hand_rgb": hand_rgb,
            "hand_back_rgb": hand_back_rgb,
            "agent_rgb": agent_rgb,
            "agent_left_rgb": agent_left_rgb,
            "agent_right_rgb": agent_right_rgb,
            
        })
        return task_obs

    def get_camera_observation(self,camera,depth=False):
        """acquire camera observation data"""
        try:
            if depth:
                # acquire an RGB image
                depth_data = camera.get_depth() # acquire a depth image
                return depth_data
            else:
                rgb_data = camera.get_rgb() # extract the RGB channels, discarding the alpha channel
                return rgb_data
        except Exception as e:
            print(f"An error occurred while acquiring camera data image: {e}")

    def get_info(self):
        # return info about current task
        return self.task.get_description()
    
    def track_forces(self):
        # track forces of the robot
        import matplotlib.pyplot as plt

        joint_force_torque = self.franka.get_measured_joint_forces()

        time_steps = np.arange(joint_force_torque.shape[0])
        labels = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']
        
        plt.figure(figsize=(12, 10))
        for i in range(6):
            plt.subplot(3, 2, i+1)
            plt.plot(time_steps, joint_force_torque[:, i], label=labels[i])
            plt.title(labels[i])
            plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def human_control_logging(self,tasks,scene):
        # this used for human control logging data to replay robot control
        # just log important data for replay
        joints_state = self.franka.get_joints_state()
        data = {
                "episode_steps": self.episode_steps,
                "actions": self.control_action.tolist(),
                "robot_states": {
                    'joint_positions': joints_state.positions[:7].tolist(),
                },
            }
        return data

    def logging_save(self,):
        # save logging data to json file
        os.makedirs(self.data_path, exist_ok=True)
        logging_time = time.time()
        self.logging_path = f"{self.data_path}/{logging_time}_episode_{self.episode_steps}.json"
        self.data_logger.save(self.logging_path)
        print(f"logging data saved to {self.logging_path}")