#!/usr/bin/env python3

import rospy
import rospkg
import os
import tf2_ros as tf2
import tf.transformations as tft
from geometry_msgs.msg import Pose
from std_msgs.msg import Empty, String
from gazebo_msgs.srv import SpawnModel, DeleteModel, ApplyBodyWrench, GetModelState
from geometry_msgs.msg import Wrench
import numpy as np
from tank_robot.msg import ShootCommand

class Shooter:
    def __init__(self):
        rospy.init_node('shooter')

        # Load the projectile model
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('tank_robot')
        self.projectile_path = os.path.join(
            rospy.get_param("~projectile_path", f"{package_path}/urdf/projectile.urdf")
        )

        with open(self.projectile_path, 'r') as f:
            self.projectile_model = f.read()

        self.spawn_srv = rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel)
        self.delete_srv = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        self.apply_wrench = rospy.ServiceProxy('/gazebo/apply_body_wrench', ApplyBodyWrench)
        self.get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)

        rospy.wait_for_service('/gazebo/spawn_urdf_model')
        rospy.wait_for_service('/gazebo/delete_model')
        rospy.wait_for_service('/gazebo/get_model_state')

        self.shoot_sub = rospy.Subscriber('/gun_shoot', ShootCommand, self.shoot_callback)
        self.shooting_done_pub = rospy.Publisher('/shooting_done', String, queue_size=1)
    
        self.tfBuffer = tf2.Buffer()
        self.listener = tf2.TransformListener(self.tfBuffer)

        self.projectile_velocity = rospy.get_param("~projectile_velocity", 10.0)
        self.gravity_timeout = rospy.get_param("~gravity_timeout", 10.0)  # Timeout in seconds
        self.retry_limit = rospy.get_param("~retry_limit", 3)
        self.collision_distance = rospy.get_param("~collision_distance", 0.5)  # Collision threshold distance
        self.gun_frame = rospy.get_param("~gun_frame", "gun_tip_link")  # Frame at gun's tip
        self.projectile_mass = rospy.get_param("~projectile_mass", 0.01)

    def shoot_callback(self, msg):
        target_model_name = msg.target_model_name
        target_position = msg.target_position
        gun_position, _ = self.get_gun_tip_pose()
        pitch = msg.pitch
        velocity = msg.velocity if msg.velocity > 0.0 else self.projectile_velocity
        timeout = msg.t + 5.0
        tries = 0

        while tries < self.retry_limit:
            tries += 1
            projectile_name = self.shoot_projectile(velocity)
            if not target_model_name:
                break
            
            collision_detected = self.wait_for_collision(projectile_name, target_model_name, timeout)

            if collision_detected:
                rospy.loginfo(f"Collision detected with {target_model_name}.")
                self.shooting_done_pub.publish(String(f"{target_model_name}, {'ELIMINATED'}"))
                self.delete_model(target_model_name)
                break
            else: 
                velocity = self.recalculate_velocity(pitch=pitch, start=gun_position, target=target_position)
                if velocity is None:
                    rospy.logerr("Failed to recalculate velocity.")
                    break

            rospy.logwarn(f"Attempt {tries} failed. Retrying...")

        if tries >= self.retry_limit:
            rospy.logerr("Failed to hit the target after maximum retries.")
            self.shooting_done_pub.publish(String(f"{target_model_name}, {'FAILED'}"))

    
    def get_gun_tip_pose(self):
        rate = rospy.Rate(10)  # 50 Hz
        
        while not rospy.is_shutdown():
            try:
                # Lookup transform from world to gun_tip_link
                transform = self.tfBuffer.lookup_transform(
                    target_frame="world", 
                    source_frame="gun_tip_link", 
                    time=rospy.Time(0),  # Use latest transform available
                    timeout=rospy.Duration(1.0)
                )
                
                # Extract position
                translation = transform.transform.translation
                position = (translation.x, translation.y, translation.z)
                # Extract orientation (quaternion)
                rotation = transform.transform.rotation
                orientation = (rotation.x, rotation.y, rotation.z, rotation.w)

                return position, orientation
                
            except tf2.LookupException:
                rospy.logwarn("Transform not available yet.")
            except tf2.ExtrapolationException as e:
                rospy.logwarn(f"Extrapolation error: {e}")
            except tf2.ConnectivityException as e:
                rospy.logwarn(f"TF connectivity error: {e}")
            
            rate.sleep()


    def shoot_projectile(self, velocity):
        try:
            gun_trans, gun_rot = self.get_gun_tip_pose()

            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = gun_trans
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = gun_rot
            wrench = Wrench()
            wrench.force.x, wrench.force.y, wrench.force.z = self.calculate_velocity(gun_rot, velocity)
            projectile_name = f"projectile_{rospy.Time.now().to_sec()}"
            self.spawn_srv(projectile_name, self.projectile_model, "", pose, "world")

            self.apply_wrench(
                body_name=f"{projectile_name}::projectile_link",
                reference_frame="world",
                wrench=wrench,
                start_time=rospy.Time(0),
                duration=rospy.Duration(self.projectile_mass)
            )

            rospy.Timer(rospy.Duration(self.gravity_timeout), lambda _: self.delete_model(projectile_name),
                        oneshot=True)

            return projectile_name

        except Exception as e:
            rospy.logwarn(f"Failed to shoot projectile: {e}")
            return None

    def wait_for_collision(self, projectile_name, target_model_name, timeout=5.0):
        start_time = rospy.Time.now()
        rate = rospy.Rate(10)  # Checking frequency

        while (rospy.Time.now() - start_time).to_sec() < timeout:
            try:
                projectile_state = self.get_model_state(projectile_name, "world")
                target_state = self.get_model_state(target_model_name, "world")

                if not projectile_state.success or not target_state.success:
                    rospy.logwarn("Failed to get model states.")
                    break

                projectile_position = np.array([
                    projectile_state.pose.position.x,
                    projectile_state.pose.position.y,
                    projectile_state.pose.position.z
                ])

                target_position = np.array([
                    target_state.pose.position.x,
                    target_state.pose.position.y,
                    target_state.pose.position.z
                ])

                # Distance calculation
                distance = np.linalg.norm(projectile_position - target_position)

                if distance < self.collision_distance:
                    return True, distance, 0
                
                # Angle calculation
                direction_to_target = target_position - projectile_position
                shooting_direction = np.dot(
                    tft.quaternion_matrix([projectile_state.pose.orientation.x,
                                        projectile_state.pose.orientation.y,
                                        projectile_state.pose.orientation.z,
                                        projectile_state.pose.orientation.w])[:3, :3],
                    np.array([1, 0, 0])  # Local x-axis is the forward direction
                )
                angle = np.arccos(
                    np.clip(
                        np.dot(direction_to_target, shooting_direction) /
                        (np.linalg.norm(direction_to_target) * np.linalg.norm(shooting_direction)),
                        -1.0,
                        1.0
                    )
                )

            except rospy.ServiceException as e:
                rospy.logwarn(f"Error checking collision: {e}")

            rate.sleep()

        return False
    
    def calculate_wrench_components(self, orientation, velocity_scalar, max_force=50):
        """
        Calculate force and duration required to set projectile velocity.
        
        Parameters:
            orientation (list): Quaternion representing orientation [x, y, z, w].
            velocity_scalar (float): Target velocity magnitude.
            projectile_mass (float): Projectile's mass in kg.
            max_force (float): Maximum allowable force magnitude.
        
        Returns:
            tuple: Force components (Fx, Fy, Fz) and duration in seconds.
        """
        # Convert quaternion to rotation matrix
        rotation_matrix = tft.quaternion_matrix(orientation)[:3, :3]

        # Local shooting direction (projectile forward direction is x-axis in local frame)
        local_direction = np.array([1, 0, 0])

        # Transform to global frame
        global_direction = np.dot(rotation_matrix, local_direction)
        global_direction_normalized = global_direction / np.linalg.norm(global_direction)

        # Calculate velocity components
        velocity_components = global_direction_normalized * velocity_scalar

        # Calculate required force and duration
        force_components = velocity_components * self.projectile_mass
        force_magnitude = np.linalg.norm(force_components)

        if force_magnitude > max_force:
            raise ValueError("Force exceeds the maximum allowable force.")

        # Calculate duration
        duration = velocity_scalar / (force_magnitude / self.projectile_mass)

        return force_components, duration

    def calculate_velocity(self, orientation, velocity):
        orientation_matrix = tft.quaternion_matrix(orientation)
        local_direction = [1, 0, 0]

        global_velocity = [
            orientation_matrix[0][0] * local_direction[0] + orientation_matrix[0][1] * local_direction[1] +
            orientation_matrix[0][2] * local_direction[2],
            orientation_matrix[1][0] * local_direction[0] + orientation_matrix[1][1] * local_direction[1] +
            orientation_matrix[1][2] * local_direction[2],
            orientation_matrix[2][0] * local_direction[0] + orientation_matrix[2][1] * local_direction[1] +
            orientation_matrix[2][2] * local_direction[2],
        ]

        return [v * velocity for v in global_velocity]
    
    def recalculate_velocity(self, pitch, start, target, gravity=9.81):
        dx, dy, dz = np.array(target) - np.array(start)

        # equation of initial velocity given pitch and distance data for ballistic motion
        dxy = np.sqrt(dx ** 2 + dy ** 2)
        denominator = (2 * np.cos(pitch) ** 2 * (dxy * np.tan(pitch) - dz))
        if denominator <= 0 or np.isnan(denominator):
            return None
        v0 = np.sqrt((gravity * dxy ** 2) / denominator)
        rospy.loginfo(f"adjusted {v0=}")

        return v0


    def delete_model(self, name):
        try:
            self.delete_srv(name)
        except rospy.ServiceException as e:
            rospy.logwarn(f"Failed to delete model {name}: {e}")

if __name__ == "__main__":
    try:
        shooter = Shooter()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
