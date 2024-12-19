#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Pose2D
import math

def send_goals():
    rospy.init_node('goal_sender', anonymous=True)

    # Publishers
    tank_goal_pub = rospy.Publisher('/tank_goal', Pose2D, queue_size=1)
    gun_goal_pub = rospy.Publisher('/gun_goal', Pose2D, queue_size=1)

    rospy.sleep(1)  # Wait for connections to establish

    # Tank goal
    tank_goal = Pose2D()
    tank_goal.x = 0.0  # Move to x=5
    tank_goal.y = -2.0  # Move to y=3
    tank_goal.theta = math.pi  # Orient to 0.5 radians
    rospy.loginfo("Sending tank goal: %s", tank_goal)
    # tank_goal_pub.publish(tank_goal)

    rospy.sleep(2)  # Wait for tank goal to be processed

    # Gun goal
    gun_goal = Pose2D()
    gun_goal.x = -math.pi/4  # Set pitch to 0.3 radians
    gun_goal.y = -math.pi/2  # Set yaw to -0.2 radians
    rospy.loginfo("Sending gun goal: %s", gun_goal)
    gun_goal_pub.publish(gun_goal)

if __name__ == '__main__':
    try:
        send_goals()
    except rospy.ROSInterruptException:
        rospy.loginfo("Goal sender node terminated.")
