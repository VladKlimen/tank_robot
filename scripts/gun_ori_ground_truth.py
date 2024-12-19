#!/usr/bin/env python3

import rospy
from gazebo_msgs.srv import GetLinkState
from geometry_msgs.msg import Pose2D
from tf.transformations import euler_from_quaternion

class GunOrientationPublisher:
    def __init__(self):
        rospy.init_node("gun_orientation_publisher")

        # Publisher for gun orientation
        self.orientation_pub = rospy.Publisher("/gun_orientation", Pose2D, queue_size=10)

        # Gazebo GetLinkState service
        rospy.wait_for_service("/gazebo/get_link_state")
        self.get_link_state = rospy.ServiceProxy("/gazebo/get_link_state", GetLinkState)

        # Parameters
        self.tank_model_name = rospy.get_param("~tank_model_name", "tank")  # Name of the tank model
        self.gun_name = rospy.get_param("~gun_name", "gun")  # Link name
        self.world_frame = rospy.get_param("~world_frame", "world")  # Reference frame

        # Publish orientation at regular intervals
        self.rate = rospy.Rate(50)  # 10 Hz

    def get_gun_orientation(self):
        try:
            # Call the Gazebo service to get the gun tip link's state
            target_name = f"{self.tank_model_name}::{self.gun_name}"
            response = self.get_link_state(target_name, self.world_frame)
            if not response.success:
                rospy.logwarn(f"Failed to get state for {target_name} from Gazebo.")
                return None

            # Extract orientation as quaternion
            orientation = response.link_state.pose.orientation
            quaternion = [orientation.x, orientation.y, orientation.z, orientation.w]

            # Convert quaternion to roll, pitch, yaw
            _, pitch, yaw = euler_from_quaternion(quaternion)
            return pitch, yaw

        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
            return None

    def publish_orientation(self):
        while not rospy.is_shutdown():
            orientation = self.get_gun_orientation()
            if orientation is not None:
                pitch, yaw = orientation

                # Publish as a Pose2D message
                msg = Pose2D()
                msg.x = pitch
                msg.y = yaw
                msg.theta = 0.0  # Unused, set to 0
                self.orientation_pub.publish(msg)

                # rospy.loginfo(f"Published gun orientation: pitch={pitch}, yaw={yaw}")

            self.rate.sleep()

if __name__ == "__main__":
    try:
        gun_orientation_publisher = GunOrientationPublisher()
        gun_orientation_publisher.publish_orientation()
    except rospy.ROSInterruptException:
        pass
