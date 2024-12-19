#!/usr/bin/env python3

from __future__ import print_function
import threading
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64, Empty
import sys
from select import select
import time
import math
from tank_robot.msg import ShootCommand

if sys.platform == 'win32':
    import msvcrt
else:
    import termios
    import tty

TwistMsg = Twist

msg = """
Control the Tank and Gun
---------------------------
Movement:
    q   w   e
    a       d
    z   s   c

W/S : increase/decrease linear speed
E/Q : increase/decrease angular speed

Gun control:
        i
    j       l
        k

Space: reset gun position
CTRL-C to quit
"""

moveBindings = {
    'w': (1, 0, 0, 0),
    's': (-1, 0, 0, 0),
    'a': (0, 0, 0, 1),
    'd': (0, 0, 0, -1),
    'q': (1, 0, 0, 1),
    'e': (1, 0, 0, -1),
    'z': (-1, 0, 0, 1),
    'c': (-1, 0, 0, -1),
}

gunBindings = {
    'i': (-1, 0),   # Pitch up
    'k': (1, 0),    # Pitch down
    'j': (0, 1),    # Yaw left
    'l': (0, -1),   # Yaw right
    ' ': (0, 0),    # Reset
    'f': (0, 0)       # Fire projectile
}

speedBindings = {
    'W': (1.1, 1),
    'S': (0.9, 1),
    'E': (1, 1.1),
    'Q': (1, 0.9),
}


class ControllerThread(threading.Thread):
    def __init__(self, name, publisher, rate=30):
        super(ControllerThread, self).__init__()
        self.name = name
        self.publisher = publisher
        self.command = None
        self.lock = threading.Lock()
        self.rate = rospy.Rate(rate)
        self.running = True
        self.start()

    def wait_for_subscribers(self):
        i = 0
        while not rospy.is_shutdown() and self.publisher.get_num_connections() == 0:
            if i == 4:
                print("Waiting for subscriber to connect to {}".format(self.publisher.name))
            rospy.sleep(0.5)
            i += 1
            i = i % 5
        if rospy.is_shutdown():
            raise Exception("Got shutdown request before subscribers connected")

    def update_command(self, command):
        with self.lock:
            self.command = command

    def stop(self):
        self.running = False
        self.join()

    def run(self):
        while self.running and not rospy.is_shutdown():
            with self.lock:
                command = self.command
            if command is not None:
                self.publisher.publish(command)
            self.rate.sleep()


def get_key(settings, timeout):
    if sys.platform == 'win32':
        return msvcrt.getwch()
    else:
        tty.setraw(sys.stdin.fileno())
        try:
            rlist, _, _ = select([sys.stdin], [], [], timeout)
            if rlist:
                return sys.stdin.read(1)
            else:
                return ''
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)


if __name__ == "__main__":
    settings = termios.tcgetattr(sys.stdin) if sys.platform != 'win32' else None

    rospy.init_node('teleop_robot_gun')

    speed = rospy.get_param("~speed", 0.5)
    turn = rospy.get_param("~turn", math.pi)
    speed_limit = rospy.get_param("~speed_limit", 1000)
    turn_limit = rospy.get_param("~turn_limit", 1000)
    repeat = rospy.get_param("~repeat_rate", 0.0)
    key_timeout = rospy.get_param("~key_timeout", 0.5)
    twist_frame = rospy.get_param("~frame_id", '')

    pitch = 0.0
    yaw = 0.0
    pitch_step = math.pi / 180
    yaw_step = math.pi / 180
    pitch_min = -math.pi/4
    # pitch_max = math.pi*5/36
    pitch_max = math.pi/6

    cmd_vel_pub = rospy.Publisher('/diff_drive_controller/cmd_vel', Twist, queue_size=1)
    gun_pitch_pub = rospy.Publisher('/gun_controller/command', Float64, queue_size=1)
    gun_yaw_pub = rospy.Publisher('/turret_controller/command', Float64, queue_size=1)
    shoot_pub = rospy.Publisher('/gun_shoot', ShootCommand, queue_size=1)

    cmd_vel_thread = ControllerThread("cmd_vel_thread", cmd_vel_pub)
    cmd_vel_thread.wait_for_subscribers()
    gun_pitch_thread = ControllerThread("gun_pitch_thread", gun_pitch_pub)
    gun_yaw_thread = ControllerThread("gun_yaw_thread", gun_yaw_pub)
    
    try:
        print(msg)
        stop_timer = time.time()  # Timer to track when the tank should stop
        stop_timeout = 0.1  # Stop the tank if no key is pressed for stop_timeout seconds

        while not rospy.is_shutdown():
            key = get_key(settings, 0.1)

            if key in moveBindings.keys():
                stop_timer = time.time()  # Reset the stop timer
                x, y, z, th = moveBindings[key]
                twist = Twist()
                twist.linear.x = x * speed
                twist.linear.y = y * speed
                twist.linear.z = z * speed
                twist.angular.z = th * turn
                cmd_vel_thread.update_command(twist)
            elif key in gunBindings.keys():
                pitch_change, yaw_change = gunBindings[key]
                if key == ' ':
                    pitch = 0.0
                    yaw = 0.0
                elif key == 'f':
                    shoot_pub.publish(ShootCommand())
                else:
                    pitch = max(min(pitch + pitch_change * pitch_step, pitch_max), pitch_min)
                    yaw += yaw_change * yaw_step
                gun_pitch_thread.update_command(Float64(pitch))
                gun_yaw_thread.update_command(Float64(yaw))
            elif key in speedBindings.keys():
                speed = min(1000, speed * speedBindings[key][0])
                turn = min(1000, turn * speedBindings[key][1])
                print(f"Speed: {speed}, Turn: {turn}")
            else:
                if key == '\x03':  # Ctrl-C to exit
                    break

            # Check if stop timeout has elapsed
            if time.time() - stop_timer > stop_timeout: 
                cmd_vel_thread.update_command(Twist()) # Stop the robot

    except Exception as e:
        print(e)
    finally:
        cmd_vel_thread.stop()
        gun_pitch_thread.stop()
        gun_yaw_thread.stop()
        if settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
