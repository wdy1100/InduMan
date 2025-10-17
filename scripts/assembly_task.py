"""
    This file defines a custom assembly task for Isaac Sim.
    It will be called by gym environment.
"""
import numpy as np

from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.api.scenes.scene import Scene
from isaacsim.core.utils.stage import add_reference_to_stage, get_stage_units
from isaacsim.core.prims import GeometryPrim,RigidPrim
from isaacsim.core.utils.rotations import euler_angles_to_quat,quat_to_euler_angles
from isaacsim.core.api.objects import FixedCuboid

class AssemblyTask(BaseTask):
    """define a custom task class that inherits from the BaseTask class"""
    
    def __init__(self, 
                 name: str, 
                 task_message = ["task_message"],
                 is_replay = False,
                 table_height = 1.03,
                 assets_root_path=None,
                 objects_to_manipulate_dict = None,
                 objects_to_interact_dict = None,
                 support = None,
                 offset=None):
        super().__init__(name, offset)
        self.task_message = task_message
        self.is_replay = is_replay # check if it is replaying the task
        self.table_height = table_height
        self.objects_to_manipulate_dict = objects_to_manipulate_dict
        self.objects_to_interact_dict = objects_to_interact_dict
        self.assets_root_path = assets_root_path
        self.support = support # support of manipulated objects, a dict
        if self.assets_root_path is None:
            raise ValueError("Could not find Isaac Sim assets folder. Please check your installation.")

    def set_up_scene(self, scene: Scene):
        """add assets to the stage and set up the task objects"""
        super().set_up_scene(scene)
        # add the objects to interact with
        self.objects_to_interact = {}
        last_object_pos = None
        last_object_ori = None
        for obj_key,obj_info in self.objects_to_interact_dict.items():
            # add the object 
            object_usd_path = obj_info['usd_path']
            add_reference_to_stage(prim_path=f"/World/{obj_info['name']}",
                                   usd_path=f"{self.assets_root_path}{object_usd_path}")
            # generate random position for the object
            object_pos_xy = np.random.uniform(high=obj_info['area'][0], low=obj_info['area'][1])
            object_pos = np.concatenate([object_pos_xy, np.array([self.table_height + obj_info['height']])])
            object_pos[0] = object_pos[0] + obj_info['offset'][0]  # apply offset xyz
            object_pos[1] = object_pos[1] + obj_info['offset'][1]
            object_pos[2] = object_pos[2] + obj_info['offset'][2]
            # generate random orientation for the object
            object_ori = np.random.uniform(high=obj_info['orientation'][0], low=obj_info['orientation'][1])
            object_orientation = euler_angles_to_quat(object_ori,degrees=True,extrinsic=False)
            
            # check if the object is the first one
            if last_object_pos is not None:
                # second object position has a offset to the first object , their orientations keep the same
                object_pos[0] = last_object_pos[0] + obj_info['offset'][0]
                object_pos[1] = last_object_pos[1] + obj_info['offset'][1]
                object_pos[2] = last_object_pos[2] + obj_info['offset'][2]
                object_orientation = last_object_ori

            last_object_pos = object_pos
            last_object_ori = object_orientation
            # create the object prim
            object_ = RigidPrim(prim_paths_expr=f"/World/{obj_info['name']}",
                                name=obj_info['name'],
                                positions=np.round(np.array([object_pos]),5)/get_stage_units(),
                                orientations=np.round(np.array([object_orientation]),4),
                                masses=np.array([obj_info['mass']]),
                                track_contact_forces=obj_info['track_contact_forces'],
                                )
            scene.add(object_)
            self.objects_to_interact[obj_info['name']] = object_ # use a dict to store the objects to interact with
        
        # add the objects to manipulate
        self.objects_support = {}
        self.objects_to_manipulate = {}
        last_object_pos = None
        last_object_ori = None
        for obj_key,obj_info in self.objects_to_manipulate_dict.items():
            if self.support['need']:
                # add support components
                support_height = self.support['height']  # the height of the support component, which is configurable within the obj_info object
                support_pos_xy = np.random.uniform(high=obj_info['area'][0], low=obj_info['area'][1])
                support_pos = np.concatenate([support_pos_xy, np.array([self.table_height + support_height / 2])])
                object_ori = np.random.uniform(high=obj_info['orientation'][0], low=obj_info['orientation'][1])
                object_orientation = euler_angles_to_quat(object_ori, degrees=True, extrinsic=False)
                if last_object_pos is not None:
                    support_pos[0] = last_object_pos[0] + obj_info['offset'][0]
                    support_pos[1] = last_object_pos[1] + obj_info['offset'][1]
                    support_pos[2] = last_object_pos[2] + obj_info['offset'][2]
                    object_orientation = last_object_ori
                last_object_pos = object_pos
                last_object_ori = object_orientation

                obj_support = FixedCuboid(
                    prim_path=f"/World/{obj_info['name']}_support",
                    name=f"{obj_info['name']}_support",
                    size=self.support['size'],  # dimensions of the support component
                    position=support_pos / get_stage_units(),
                    orientation=object_orientation,
                    scale=self.support['scale'],  # scaling factor of the support component
                    visible=True,
                )
                self.objects_support[f"{obj_info['name']}_support"] = obj_support
                # add an object, positioning it on the support component
                object_usd_path = obj_info['usd_path']
                add_reference_to_stage(
                    prim_path=f"/World/{obj_info['name']}",
                    usd_path=f"{self.assets_root_path}{object_usd_path}"
                )
                object_pos = np.concatenate([support_pos_xy, np.array([self.table_height + support_height + obj_info['height']])])
                object_ = RigidPrim(
                    prim_paths_expr=f"/World/{obj_info['name']}",
                    name=obj_info['name'],
                    positions=np.round(np.array([object_pos]),5) / get_stage_units(),
                    orientations=np.round(np.array([object_orientation]),4),
                    masses=np.array([obj_info['mass']]),
                    contact_filter_prim_paths_expr=[f"/World/{obj_info['interact_object']}"],
                )
                scene.add(obj_support)
                scene.add(object_)
                self.objects_to_manipulate[obj_info['name']] = object_
            else:
                object_usd_path = obj_info['usd_path']
                add_reference_to_stage(prim_path=f"/World/{obj_info['name']}",
                                    usd_path=f"{self.assets_root_path}{object_usd_path}")
                # generate random position for the object
                object_pos_xy = np.random.uniform(high=obj_info['area'][0], low=obj_info['area'][1])
                object_pos = np.concatenate([object_pos_xy, np.array([self.table_height + obj_info['height']])])
                # generate random orientation for the object
                object_ori = np.random.uniform(high=obj_info['orientation'][0], low=obj_info['orientation'][1])
                object_orientation = euler_angles_to_quat(object_ori,degrees=True,extrinsic=False)
                # check if the object is the first one
                if last_object_pos is not None:
                    # second object position has a offset to the first object , their orientations keep the same
                    object_pos[0] = last_object_pos[0] + obj_info['offset'][0]
                    object_pos[1] = last_object_pos[1] + obj_info['offset'][1]
                    object_pos[2] = last_object_pos[2] + obj_info['offset'][2]
                    object_orientation = last_object_ori

                last_object_pos = object_pos
                last_object_ori = object_orientation
                # create the object prim
                object_ = RigidPrim(prim_paths_expr=f"/World/{obj_info['name']}",
                                    name=obj_info['name'],
                                    positions=np.round(np.array([object_pos]),5)/get_stage_units(),
                                    orientations=np.round(np.array([object_orientation]),4),
                                    masses=np.array([obj_info['mass']]),
                                    contact_filter_prim_paths_expr=[f"/World/{obj_info['interact_object']}"],
                                    ) 
                scene.add(object_)
                self.objects_to_manipulate[obj_info['name']] = object_ # use a dict to store the objects to manipulate
        
    def get_manipulated_objects(self,only_names=False):
        """return the list of objects to manipulate
            default return the object prims"""
        if only_names:
            return self.objects_to_manipulate.keys()
        else:
            return self.objects_to_manipulate

    def get_interact_objects(self,only_names=False):
        """return the list of objects to interact with
            default return the object prims"""
        if only_names:
            return self.objects_to_interact.keys()
        else:
            return self.objects_to_interact
    
    def get_task_objects(self):
        """return the dict of all objects in the task"""
        task_objects= dict()
        for obj_name, obj in self.objects_to_manipulate.items():
            task_objects[obj_name] = obj
        for obj_name, obj in self.objects_to_interact.items():
            task_objects[obj_name] = obj
        return task_objects

    def reset_objects_position(self):
        """reset the task by changing the position of the objects"""
        if not self.is_replay:
            # reset the position of the interact objects
            last_object_pos = None
            last_object_ori = None
            for obj_key,obj_info in self.objects_to_interact_dict.items():
                object_pos_xy = np.random.uniform(high=obj_info['area'][0], low=obj_info['area'][1])
                object_pos = np.concatenate([object_pos_xy, np.array([self.table_height + obj_info['height']])])
                object_pos[0] = object_pos[0] + obj_info['offset'] [0] # apply offset in x axis
                object_pos[1] = object_pos[1] + obj_info['offset'][1]
                object_pos[2] = object_pos[2] + obj_info['offset'][2]
                obj = self.get_task_objects()[obj_info['name']]
                # get object orientation
                object_ori = np.random.uniform(high=obj_info['orientation'][0], low=obj_info['orientation'][1])
                object_orientation = euler_angles_to_quat(object_ori,degrees=True,extrinsic=False)
                # check if the object is the first one
                if last_object_pos is not None:
                    # second object position has a offset to the first object , their orientations keep the same
                    object_pos[0] = last_object_pos[0] + obj_info['offset'][0]
                    object_pos[1] = last_object_pos[1] + obj_info['offset'][1]
                    object_pos[2] = last_object_pos[2] + obj_info['offset'][2]
                    object_orientation = last_object_ori
                    
                last_object_pos = object_pos
                last_object_ori = object_orientation
                # set object position and orientation
                obj.set_world_poses(positions=np.round(np.array([object_pos]),5)/get_stage_units(),
                                    orientations=np.round(np.array([object_orientation]),4))
            
            # reset the position of the manipulate objects
            last_object_pos = None
            last_object_ori = None
            for obj_key,obj_info in self.objects_to_manipulate_dict.items():
                if self.support['need']:
                    support_height = self.support['height']  # the height of the support component, which is configurable within the obj_info object
                    support_pos_xy = np.random.uniform(high=obj_info['area'][0], low=obj_info['area'][1])
                    support_pos = np.concatenate([support_pos_xy, np.array([self.table_height + support_height / 2])])
                    object_ori = np.random.uniform(high=obj_info['orientation'][0], low=obj_info['orientation'][1])
                    object_orientation = euler_angles_to_quat(object_ori, degrees=True, extrinsic=False)
                    if last_object_pos is not None:
                        support_pos[0] = last_object_pos[0] + obj_info['offset'][0]
                        support_pos[1] = last_object_pos[1] + obj_info['offset'][1]
                        support_pos[2] = last_object_pos[2] + obj_info['offset'][2]
                        object_orientation = last_object_ori
                    last_object_pos = object_pos
                    last_object_ori = object_orientation

                    obj_support = self.objects_support[f"{obj_info['name']}_support"]
                    obj_support.set_world_pose(position=support_pos/get_stage_units(),
                                                orientation=object_orientation)
                    obj = self.get_task_objects()[obj_info['name']]
                    # get object position and orientation
                    object_pos = np.concatenate([support_pos_xy, np.array([self.table_height + support_height + obj_info['height']])])
                    object_ = self.objects_to_manipulate[obj_info['name']]
                    object_.set_world_poses(positions=np.round(np.array([object_pos]),5)/get_stage_units(),
                                            orientations=np.round(np.array([object_orientation]),4))
                
                else:
                    object_pos_xy = np.random.uniform(high=obj_info['area'][0], low=obj_info['area'][1])
                    object_pos = np.concatenate([object_pos_xy, np.array([self.table_height + obj_info['height']])])
                    obj = self.get_task_objects()[obj_info['name']]
                    # get object orientation
                    object_ori = np.random.uniform(high=obj_info['orientation'][0], low=obj_info['orientation'][1])
                    object_orientation = euler_angles_to_quat(object_ori,degrees=True,extrinsic=False)
                    # check if the object is the first one
                    if last_object_pos is not None:
                        # second object position has a offset to the first object , their orientations keep the same
                        object_pos[0] = last_object_pos[0] + obj_info['offset'][0]
                        object_pos[1] = last_object_pos[1] + obj_info['offset'][1]
                        object_pos[2] = last_object_pos[2] + obj_info['offset'][2]
                        object_orientation = last_object_ori

                    last_object_pos = object_pos
                    last_object_ori = object_orientation
                    # set object position and orientation
                    obj.set_world_poses(positions=np.round(np.array([object_pos]),5)/get_stage_units(),
                                        orientations=np.round(np.array([object_orientation]),4))
            
    def is_done(self):
        """check if the task is done"""
        for obj_key,obj_info in self.objects_to_manipulate_dict.items():
            obj = self.get_task_objects()[obj_info['name']]
            interact_obj = self.get_task_objects()[obj_info['interact_object']]
            obj_pos = obj.get_world_poses()[0][0]
            interact_obj_pos = interact_obj.get_world_poses()[0][0]
            thresholds = obj_info['success_criteria'][0] # position thresholds

            # until reach some place position, the task is not done
            position_error = (
                np.abs(obj_pos[0] - interact_obj_pos[0]) < thresholds[0] and
                np.abs(obj_pos[1] - interact_obj_pos[1]) < thresholds[1] and
                np.abs(obj_pos[2] - interact_obj_pos[2]) < thresholds[2]
            )
            
            # until reach some force, the task is not done
            force_error = True if any(np.abs(obj.get_contact_force_matrix(None,dt=1./60)[0][0]) > obj_info['success_criteria'][1][0]) else False

            if not (position_error and force_error):
                return False
        
        # if all objects are successful, the task is done
        return True
            
    def get_params(self):
        """return the task parameters"""
        params_representation = dict()
        params_representation['task_name'] = {"value": self.name,"modifiable": False}
        params_representation['objects_to_manipulate'] = {"value": self.get_manipulated_objects(only_names=True),"modifiable": False}
        params_representation['objects_to_interact'] = {"value": self.get_interact_objects(only_names=True),"modifiable": False}
        params_representation['table_height'] = {"value": self.table_height,"modifiable": True}
        return params_representation
    
    def get_description(self):
        "get the task description"
        description = dict()
        description['task_name'] = self.name
        description['objects_to_manipulate'] = self.get_manipulated_objects(only_names=True)
        description['objects_to_interact'] = self.get_interact_objects(only_names=True)
        description['task_message'] = self.task_message[0]
        return description
    
    def get_observations(self,with_forces=False):
        """return the observations of the task"""
        observations = dict()
        objects = self.get_manipulated_objects()
        for obj_name, obj in objects.items():
            position, orientation = obj.get_world_poses()
            pos = np.round(np.asarray(position).reshape(-1), 5)
            ori = np.round(np.asarray(orientation).reshape(-1),4)
            observations[obj.name + "_pose"] = np.concatenate([pos, ori],axis=0)
            if isinstance(obj, RigidPrim):
                if with_forces:
                    contact_forces = obj.get_contact_force_matrix(None, dt=1./60)[0][0]
                    observations[obj.name + "_contact_forces"] = contact_forces
        for obj_name, obj in self.get_interact_objects().items():
            position, orientation = obj.get_world_poses()
            pos = np.round(np.asarray(position).reshape(-1), 5)
            ori = np.round(np.asarray(orientation).reshape(-1),4)
            observations[obj.name + "_pose"] = np.concatenate([pos, ori],axis=0)
        return observations
    
