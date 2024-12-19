#!/usr/bin/env python3

import rospy
import os
import random
import subprocess
import tempfile
from gazebo_msgs.srv import SpawnModel
from geometry_msgs.msg import Pose
from tf.transformations import quaternion_from_euler
from GridBasedPlanner import GridBasedPlanner
from Mapper import GoalStatus

class GoalsParams:
    def __init__(self, goals_parameters_path="../worlds/goals_params.txt", goal_name_prefix="cat_goal"):

        self.goal_name_prefix = goal_name_prefix
        self.goals_types = {
            "1": {"name": "laying", "size": (0.5, 0.205, 0.372), "default_scale": (1, 1, 1), "last_serial": 0},
            "2": {"name": "sleeping", "size": (0.46, 0.36, 0.178), "default_scale": (1, 1, 1), "last_serial": 0},
            "3": {"name": "standing", "size": (1.275, 0.343, 1.315), "default_scale": (0.4, 0.4, 0.4), "last_serial": 0},
            "4": {"name": "sitting", "size": (0.322, 0.205, 0.464), "default_scale": (0.8, 0.8, 0.8), "last_serial": 0}
            }
        
        self.goals_parameters_path = goals_parameters_path

        self.params = {}

    def add_goal_params(self, params:dict):
        goal_params = {}
        goal_params["type"] = str(params["type"]) #  required
        goal_params["serial"] = str(self.goals_types[goal_params["type"]]["last_serial"])
        self.goals_types[goal_params["type"]]["last_serial"] += 1
        goal_name = f'{self.goal_name_prefix}_{goal_params["type"]}_{goal_params["serial"]}'
        # position (required)
        goal_params["x"] = float(params["x"]) if "x" in params else 0.0
        goal_params["y"] = float(params["y"]) if "y" in params else 0.0
        goal_params["z"] = float(params["z"]) if "z" in params else 0.0
        # rotation (optional, default 0)
        goal_params["rx"] = float(params["rx"]) if "rx" in params else 0.0
        goal_params["ry"] = float(params["ry"]) if "ry" in params else 0.0
        goal_params["rz"] = float(params["rz"]) if "rz" in params else 0.0
        # scale
        goal_params["sx"] = float(params["sx"]) if "sx" in params else self.goals_types[goal_params["type"]]["default_scale"][0]
        goal_params["sy"] = float(params["sy"]) if "sy" in params else self.goals_types[goal_params["type"]]["default_scale"][1]
        goal_params["sz"] = float(params["sz"]) if "sz" in params else self.goals_types[goal_params["type"]]["default_scale"][2]
        # collision box
        goal_params["cx"] = round(goal_params["sx"] * self.goals_types[goal_params["type"]]["size"][0], 4)
        goal_params["cy"] = round(goal_params["sx"] * self.goals_types[goal_params["type"]]["size"][1], 4)
        goal_params["cz"] = round(goal_params["sx"] * self.goals_types[goal_params["type"]]["size"][2], 4)

        self.params[goal_name] =  goal_params
    
    def add_goals_params_from_file(self):
        try:
            with open(self.goals_parameters_path, 'r') as file:
                for line in file:
                    line = line.strip()
                    if line:
                        params = dict(item.split('=') for item in line.split())
                        self.add_goal_params(params)
        except Exception as e:
            rospy.logerr(f"Could not add goals from file: {e}")


class GoalsSpawner:
    def __init__(self, planner:GridBasedPlanner, xacro_file="../worlds/goals.sdf.xacro", output_sdf_dir="../worlds/goals_tmp",
                 goals_parameters_path="../worlds/goals_params.txt", goal_name_prefix="cat_goal", generate_sdf=True):
        
        self.planner = planner

        self.xacro_file = xacro_file
        self.output_sdf_dir = output_sdf_dir
        
        self.goals_params = GoalsParams(goals_parameters_path=goals_parameters_path, goal_name_prefix=goal_name_prefix)
        self.goals_params.add_goals_params_from_file()
        self.sdf_paths = {}
        self.spawned = []

        if generate_sdf:
            for goal_name, params in self.goals_params.params.items():
                self.generate_sdf_from_xacro(goal_name, params, for_map=False)
                self.generate_sdf_from_xacro(goal_name, params, for_map=True)

    def generate_sdf_from_xacro(self, goal_name, params:dict, for_map=True):
        """
        Generates an SDF file from a Xacro file with the provided parameters.

        Args:
            xacro_file (str): Path to the input Xacro file.
            output_sdf (str): Path to save the generated SDF file.
            goal_name (str): goal name 
            params (dict): Dictionary of parameter names and values to pass to Xacro.
        """
        os.makedirs(self.output_sdf_dir, exist_ok=True)
        sdf_path = os.path.join(self.output_sdf_dir + ("_for_map" if for_map else ""), goal_name + ".sdf")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".xacro", delete=False) as temp_xacro:
            temp_xacro_name = temp_xacro.name
            temp_xacro.write(self.generate_xacro_request(params, for_map=for_map))
    
        try:
            xacro_command = ["rosrun", "xacro", "xacro", temp_xacro_name, "-o", sdf_path]
            subprocess.check_call(xacro_command)
            if not for_map:
                self.sdf_paths[goal_name] = sdf_path
        except subprocess.CalledProcessError as e:
            rospy.logerr(f"Error while generating SDF: {e}")
        finally:
            os.remove(temp_xacro_name)

    @staticmethod
    def generate_xacro_request(params, for_map=True):
        x, y, z = (params["x"], params["y"], params["z"]) if for_map else (0, 0, 0)
        rx, ry, rz = (params["rx"], params["ry"], params["rz"]) if for_map else (0, 0, 0)
        xacro_file = f"""<?xml version="1.0"?>
        <sdf version="1.7" xmlns:xacro="http://www.ros.org/wiki/xacro">
            <xacro:include filename="$(find tank_robot)/worlds/goals.sdf.xacro"/>
            <xacro:cat type="{params["type"]}" serial="{params["serial"]}" 
                        x="{x}" y="{y}" z="{z}" 
                        rx="{rx}" ry="{ry}" rz="{rz}" 
                        sx="{params["sx"]}" sy="{params["sy"]}" sz="{params["sz"]}" 
                        cx="{params["cx"]}" cy="{params["cy"]}" cz="{params["cz"]}"/>
        </sdf>
        """
        return xacro_file

    def spawn_model_in_gazebo(self, model_name):
        """
        Spawns a model in Gazebo using the generated SDF file.

        Args:
            model_name (str): Name of the model in Gazebo.
        """
        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        try:
            spawn_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)

            with open(self.sdf_paths[model_name], 'r') as f:
                sdf_content = f.read()

            pose = Pose()
            position = (self.goals_params.params[model_name]["x"], 
                        self.goals_params.params[model_name]["y"], 
                        self.goals_params.params[model_name]["z"])
            rotation = (self.goals_params.params[model_name]["rx"], 
                        self.goals_params.params[model_name]["ry"], 
                        self.goals_params.params[model_name]["rz"])
            pose.position.x, pose.position.y, pose.position.z = position
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = quaternion_from_euler(*rotation)
           
            spawn_model(model_name=model_name, model_xml=sdf_content, robot_namespace="", initial_pose=pose, reference_frame="world")
            # rospy.loginfo(f"Model '{model_name}' spawned successfully.")
            return True
        except rospy.ServiceException as e:
            rospy.logerr(f"Failed to spawn model: {e}")
            return False

    def spawn_n_goals(self, n=1, randomize=True):
        not_spawned = [goal for goal in self.goals_params.params if goal not in self.spawned]
        spawned_now = []
        if len(not_spawned) == 0:
            rospy.loginfo("Failed to spawn n models: all models already spawned.")
            return False
        if randomize:
            keys = random.sample(not_spawned, min(n, len(not_spawned)))
        else:
            keys = not_spawned[:n]
        
        rospy.loginfo("Spawning goals...")
        for key in keys:
            if self.spawn_model_in_gazebo(key):
                self.spawned.append(key)
                spawned_now.append(key)
                self.planner.mapper.goals[key]["status"] = GoalStatus.QUEUED
        
        return spawned_now

    def reset_spawned(self):
        self.spawned = []
        
    