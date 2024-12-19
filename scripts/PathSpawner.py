#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Pose
from gazebo_msgs.srv import SpawnModel, DeleteModel
import tf.transformations as tft
import numpy as np
from queue import Queue


class PathSpawner:
    def __init__(self):
        self.spheres = Queue()
        self.arrows = Queue()
        self.last_sphere_id = 0
        self.last_arrow_id = 0

        self.spawn_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        self.delete_srv = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)

        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        rospy.wait_for_service('/gazebo/delete_model')
    
    def spawn_arrow(self, start, end, arrow_radius=0.02, arrow_color="Gazebo/Green"):
        """
        Spawns an arrow (cylinder) in Gazebo from the start to the end point.
        """
        start = np.array(start, dtype=float)
        end = np.array(end, dtype=float)
        model_id = self.last_arrow_id

        direction = end - start
        length = np.linalg.norm(direction)

        if length == 0:
            rospy.logerr("Start and end points are the same. Cannot spawn an arrow.")
            return

        direction /= length

        # Compute quaternion to align arrow with direction vector
        z_axis = np.array([0, 0, 1])  # Default Z-axis of the cylinder
        v = direction
        v_norm = np.linalg.norm(v)

        if v_norm == 0:
            rospy.logerr("Direction vector has zero length. Cannot align arrow.")
            return

        rotation_axis = np.cross(z_axis, v)
        rotation_axis_norm = np.linalg.norm(rotation_axis)
        if rotation_axis_norm != 0:
            rotation_axis /= rotation_axis_norm

        rotation_angle = np.arccos(np.clip(np.dot(z_axis, v) / v_norm, -1.0, 1.0))
        quaternion = tft.quaternion_about_axis(rotation_angle, rotation_axis)

        pose = Pose()
        pose.position.x = (start[0] + end[0]) / 2
        pose.position.y = (start[1] + end[1]) / 2
        pose.position.z = (start[2] + end[2]) / 2
        pose.orientation.x = quaternion[0]
        pose.orientation.y = quaternion[1]
        pose.orientation.z = quaternion[2]
        pose.orientation.w = quaternion[3]

        arrow_sdf = f"""
        <sdf version="1.6">
          <model name="arrow">
            <static>true</static>
            <link name="link">
              <visual name="base">
                <geometry>
                  <cylinder>
                    <radius>{arrow_radius}</radius>
                    <length>{length}</length>
                  </cylinder>
                </geometry>
                <material>
                  <script>
                    <uri>file://media/materials/scripts/gazebo.material</uri>
                    <name>{arrow_color}</name>
                  </script>
                </material>
              </visual>
            </link>
          </model>
        </sdf>
        """

        try:
            self.spawn_model(f"arrow_{model_id}", arrow_sdf, "", pose, "world")
            self.arrows.put(f"arrow_{model_id}")
            self.last_arrow_id += 1
        except rospy.ServiceException as e:
            rospy.logerr(f"Failed to spawn arrow: {e}")

    def spawn_sphere(self, position, radius=0.05, color="Gazebo/Red"):
        """
        Spawns sphere and the given position
        """
        model_id = self.last_sphere_id

        sdf_content = f"""
        <sdf version="1.6">
          <model name="sphere">
            <static>1</static>
            <link name="link">
              <visual name="visual">
                <geometry>
                  <sphere>
                    <radius>{radius}</radius>
                  </sphere>
                </geometry>
                <material>
                  <script>
                    <uri>file://media/materials/scripts/gazebo.material</uri>
                    <name>{color}</name>
                  </script>
                </material>
              </visual>
            </link>
          </model>
        </sdf>
        """

        pose = Pose()
        pose.position.x = position[0]
        pose.position.y = position[1]
        pose.position.z = position[2]
        pose.orientation.x = 0.0
        pose.orientation.y = 0.0
        pose.orientation.z = 0.0
        pose.orientation.w = 1.0

        try:
            self.spawn_model(f"sphere_{model_id}", sdf_content, "", pose, "world")
            self.spheres.put(f"sphere_{model_id}")
            self.last_sphere_id += 1
        except rospy.ServiceException as e:
            rospy.logerr(f"Failed to spawn sphere '{model_id}': {e}")

    def spawn_paths(self, paths):
        if not paths:
            return
        for path in paths:
            for i in range(len(path) - 1):
                self.spawn_sphere(path[i])
                self.spawn_arrow(path[i], path[i+1])
        self.spawn_sphere(paths[-1][-1])

    def delete_model(self, type):
        if type == "sphere" and not self.spheres.empty():
            model_name = self.spheres.get()
        elif type == "arrow" and not self.arrows.empty():
            model_name = self.arrows.get()
        try:
            self.delete_srv(model_name)
        except rospy.ServiceException as e:
            rospy.logwarn(f"Failed to delete model {model_name}: {e}")

    def delete_all(self):
        while not self.spheres.empty():
            self.delete_model("sphere")
        while not self.arrows.empty():
            self.delete_model("arrow")
        
        

