#!/usr/bin/env python3

import rospy
from std_msgs.msg import String

def publish_world_name():
    rospy.init_node('world_name_publisher')
    world_name = rospy.get_param('~world_name', 'by_small')

    # Create a publisher for the world name
    pub = rospy.Publisher('gazebo_world_name', String, queue_size=10)

    rate = rospy.Rate(1)  # 1 Hz publish rate
    while not rospy.is_shutdown():
        pub.publish(world_name)
        rate.sleep()

if __name__ == '__main__':
    try:
        publish_world_name()
    except rospy.ROSInterruptException:
        pass
