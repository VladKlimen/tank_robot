#!/usr/bin/env python3

import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped
from gazebo_msgs.msg import ModelStates
from nav_msgs.msg import Odometry

class OdomPublisher:
    def __init__(self):
        rospy.init_node('odom_publisher')

        self.robot_name = rospy.get_param('~robot_name', 'robot_model_name')
        self.odom_frame = rospy.get_param('~odom_frame', 'odom')
        self.base_frame = rospy.get_param('~base_frame', 'base_link')

        self.odom_pub = rospy.Publisher('/odom', Odometry, queue_size=10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        rospy.Subscriber('/gazebo/model_states', ModelStates, self.model_states_callback)

        self.last_time = rospy.Time(0)
        self.last_pose = None
        self.last_twist = None

    def model_states_callback(self, msg):
        if self.robot_name in msg.name:
            index = msg.name.index(self.robot_name)
            pose = msg.pose[index]
            twist = msg.twist[index]
            current_time = rospy.Time.now()

            # Check if time has advanced at all
            if current_time <= self.last_time:
                # If time hasn't advanced, skip publishing to avoid identical timestamp issues
                return

            # Check if pose or twist changed
            pose_changed = (self.last_pose is None or pose != self.last_pose)
            twist_changed = (self.last_twist is None or twist != self.last_twist)

            # Also allow publishing if time advanced, even if pose didn't change
            time_advanced = (current_time > self.last_time)

            if time_advanced or pose_changed or twist_changed:
                self.publish_odometry(pose, twist, current_time)
                self.broadcast_tf(pose, current_time)

                self.last_time = current_time
                self.last_pose = pose
                self.last_twist = twist

    def publish_odometry(self, pose, twist, time):
        odom_msg = Odometry()
        odom_msg.header.stamp = time
        odom_msg.header.frame_id = self.odom_frame
        odom_msg.child_frame_id = self.base_frame
        odom_msg.pose.pose = pose
        odom_msg.twist.twist = twist
        self.odom_pub.publish(odom_msg)

    def broadcast_tf(self, pose, time):
        tf_msg = TransformStamped()
        tf_msg.header.stamp = time
        tf_msg.header.frame_id = self.odom_frame
        tf_msg.child_frame_id = self.base_frame
        tf_msg.transform.translation.x = pose.position.x
        tf_msg.transform.translation.y = pose.position.y
        tf_msg.transform.translation.z = pose.position.z
        tf_msg.transform.rotation = pose.orientation
        self.tf_broadcaster.sendTransform(tf_msg)

if __name__ == '__main__':
    try:
        OdomPublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
