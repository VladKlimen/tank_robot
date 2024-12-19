#!/usr/bin/env python3

import rospy
from GridBasedPlanner import GridBasedPlanner
from PathSpawner import PathSpawner
from Mapper import GoalStatus
from geometry_msgs.msg import Pose2D
from std_msgs.msg import Empty, String
from tank_robot.msg import ShootCommand

class PlanExecutor:
    def __init__(self, planner: GridBasedPlanner, path_spawner: PathSpawner):
        # rospy.init_node('plan_executor', anonymous=True)

        # Publishers
        self.tank_goal_pub = rospy.Publisher('/tank_goal', Pose2D, queue_size=1)
        self.gun_goal_pub = rospy.Publisher('/gun_goal', Pose2D, queue_size=1)
        self.shoot_pub = rospy.Publisher('/gun_shoot', ShootCommand, queue_size=1)
        # Command queue and path spawner
        self.planner = planner
        self.path_spawner = path_spawner

        # Subscribers for feedback
        rospy.Subscriber('/goal_reached', Empty, self.goal_reached_callback)
        rospy.Subscriber('/shooting_done', String, self.shooting_done_callback)

        self.current_command = None
        self.command_in_progress = False

        rospy.sleep(1)  # Wait for connections to establish

    def execute_queue(self):
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            if self.planner.plan.empty():
                return True
            if not self.command_in_progress:
                self.current_command = self.planner.plan.get()
                self.command_in_progress = True
                self.execute_command(self.current_command)
            rate.sleep()
        return False

    def execute_command(self, command):
        command_type = command['type']
        params = command['params']

        if command_type == 'drive':
            goal = Pose2D(x=params['x'], y=params['y'], theta=0.0)
            # goal.x, goal.y, goal.theta = params['x'], params['y'], 0.0
            self.tank_goal_pub.publish(goal)
            self.planner.mapper.goals[params["goal_id"]]["status"] = GoalStatus.EXECUTING
            rospy.loginfo(f"Executing drive command to {goal}")

        elif command_type == 'aim':
            gun_goal = Pose2D(x=params['pitch'], y=params['yaw'], theta=params['relative'])
            self.gun_goal_pub.publish(gun_goal)
            rospy.loginfo(f"Executing aim command with pitch={params['pitch']}, yaw={params['yaw']}, relative={params['relative']}")

        elif command_type == 'shoot':
            msg = ShootCommand()
            msg.velocity = params['v0']
            msg.t = params['t']
            msg.target_model_name = params["goal_id"]
            msg.target_position = params["goal_position"]
            msg.pitch = params["pitch"]
            self.shoot_pub.publish(msg)
            rospy.loginfo(f"Executing shoot command with v0={params['v0']}")

    def goal_reached_callback(self, msg):
        if self.current_command and self.current_command['type'] in ('drive', 'aim'):
            rospy.loginfo(f"{self.current_command['type']} command completed")

            if self.current_command['type'] == 'drive':
                 # deleting the last part of the plan path
                self.path_spawner.delete_model("sphere")
                self.path_spawner.delete_model("arrow")
                

            self.command_in_progress = False

    def shooting_done_callback(self, msg):
        if self.current_command and self.current_command['type'] == 'shoot':
            rospy.loginfo("Shoot command completed")
            goal_id, status = msg.data.split(", ")
            self.planner.mapper.goals[goal_id]["status"] = GoalStatus[status]
            rospy.loginfo(f"{goal_id=}, {GoalStatus[status]=}")
            self.command_in_progress = False

