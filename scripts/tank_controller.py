#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Pose2D, Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64, Empty
import math
import time
from tf.transformations import euler_from_quaternion


class TankGoalController:
    def __init__(self):
        rospy.init_node('tank_goal_controller', anonymous=True)

        # Publishers
        self.cmd_vel_pub = rospy.Publisher('/diff_drive_controller/cmd_vel', Twist, queue_size=1)
        self.gun_pitch_pub = rospy.Publisher('/gun_controller/command', Float64, queue_size=1)
        self.gun_yaw_pub = rospy.Publisher('/turret_controller/command', Float64, queue_size=1)
        self.goal_reached_pub = rospy.Publisher('/goal_reached', Empty, queue_size=1)

        # Subscribers
        rospy.Subscriber('/tank_goal', Pose2D, self.handle_tank_goal)
        rospy.Subscriber('/gun_goal', Pose2D, self.handle_gun_goal)
        rospy.Subscriber('/odom', Odometry, self.update_odometry)
        rospy.Subscriber('/gun_orientation', Pose2D, self.update_gun_orientation)

        # Tank State
        self.tank_position = Pose2D()  # Current position and orientation

        # Gun State
        self.current_gun_pitch = None
        self.current_gun_yaw = None

        # Parameters
        self.linear_speed = rospy.get_param("~linear_speed", 0.8)
        self.max_angular_speed = rospy.get_param("~max_angular_speed", 0.8)
        self.goal_tolerance = rospy.get_param("~goal_tolerance", 0.1)
        self.angular_tolerance = rospy.get_param("~angular_tolerance", 0.005)
        self.slowdown_distance = rospy.get_param("~slowdown_distance", 1.5)

        # PID Controller for Rotation
        self.k_p = rospy.get_param("~k_p", 1.0)  # Proportional gain

    def normalize_angle(self, angle):
        """
        Normalize an angle to the range [-pi, pi].
        """
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def update_odometry(self, msg):
        """
        Update the tank's position and orientation from odometry data.
        """
        self.tank_position.x = msg.pose.pose.position.x
        self.tank_position.y = msg.pose.pose.position.y
        orientation = msg.pose.pose.orientation
        _, _, self.tank_position.theta = euler_from_quaternion([
            orientation.x, orientation.y, orientation.z, orientation.w
        ])

    def handle_tank_goal(self, goal):
        """
        Move the tank to the specified goal position and orientation.
        """
        rospy.loginfo("Received tank goal: x=%f, y=%f, theta=%f", goal.x, goal.y, goal.theta)
        rate = rospy.Rate(500)

        # Step 1: Rotate to face the goal
        while not rospy.is_shutdown():
            dx = goal.x - self.tank_position.x
            dy = goal.y - self.tank_position.y
            angle_to_goal = math.atan2(dy, dx)
            angular_error = self.normalize_angle(angle_to_goal - self.tank_position.theta)

            if abs(angular_error) < self.angular_tolerance:  # Goal direction reached
                rospy.loginfo("Tank aligned with goal direction")
                break

            # Apply PID control for angular speed
            twist = Twist()
            twist.angular.z = self.k_p * angular_error
            twist.angular.z = max(-self.max_angular_speed, min(self.max_angular_speed, 5 * twist.angular.z))  # Limit speed
            self.cmd_vel_pub.publish(twist)
            rate.sleep()

        # Stop rotation
        self.cmd_vel_pub.publish(Twist())

        # Step 2: Drive to the goal
        while not rospy.is_shutdown():
            dx = goal.x - self.tank_position.x
            dy = goal.y - self.tank_position.y
            distance = math.sqrt(dx**2 + dy**2)

            if distance < self.goal_tolerance:
                rospy.loginfo("Tank driving goal reached")
                self.goal_reached_pub.publish(Empty())
                break

            # Slow down as the tank approaches the goal
            speed = self.linear_speed
            if distance < self.slowdown_distance:
                # rospy.loginfo("Slowing down...")
                speed = distance / self.slowdown_distance

            twist = Twist()
            twist.linear.x = speed
            self.cmd_vel_pub.publish(twist)
            rate.sleep()

        # Stop the tank
        self.cmd_vel_pub.publish(Twist())

    def handle_gun_goal(self, goal):
        """
        Store the gun pitch and yaw goals in the world frame.
        """
        rospy.loginfo("Received gun goal: pitch=%f, yaw=%f", -goal.x, goal.y)
        self.rotate_gun(-goal.x, goal.y, int(goal.theta))  # Trigger gun rotation after reaching the goal

    def rotate_gun(self, pitch, yaw, relative=1):
        """
        Rotate the gun to the desired pitch and yaw with respect to the world.
        Wait until the rotation is complete before publishing to goal_reached.
        """
        if not relative:
            # Convert world yaw to yaw relative to the tank's orientation
            gun_yaw = self.normalize_angle(yaw - self.tank_position.theta)
        else:
            gun_yaw = yaw

        rospy.loginfo("Rotating gun to pitch=%f, yaw=%f", pitch, gun_yaw)
        self.gun_pitch_pub.publish(Float64(pitch))
        self.gun_yaw_pub.publish(Float64(gun_yaw))

        # Wait for gun to align
        if not relative:
            rate = rospy.Rate(50)
            while not rospy.is_shutdown():
                if self.is_gun_aligned(self.current_gun_yaw, yaw) and self.is_gun_aligned(self.current_gun_pitch, pitch):
                    time.sleep(1)
                    rospy.loginfo("Gun yaw rotation completed.")
                    break
                rate.sleep()    

        # Publish goal reached
        self.goal_reached_pub.publish(Empty())
        rospy.loginfo("Tank aiming goal reached")

    def is_gun_aligned(self, current_angle, target_angle, epsilon=0.01):
        """
        Check if the gun has reached the desired angle (pitch or yaw) within a specified tolerance.
        """
        if current_angle is None:
            rospy.logwarn("Gun current orientation not available.")
            return False

        # Compare current angle with goal angle
        diff = abs(current_angle - target_angle)
        return diff <= epsilon

    def update_gun_orientation(self, msg):
        """
        Update the current gun pitch and yaw based on feedback from the gun orientation topic.
        """
        self.current_gun_pitch = msg.x
        self.current_gun_yaw = msg.y


if __name__ == '__main__':
    try:
        controller = TankGoalController()
        rospy.loginfo("Tank Goal Controller Node Running...")
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Tank Goal Controller Node Shutting Down...")
