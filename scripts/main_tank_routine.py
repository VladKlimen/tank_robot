#!/usr/bin/env python3

import rospy
import sys
import time
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from geometry_msgs.msg import Pose2D
from PlanExecutor import PlanExecutor
from Mapper import GazeboWaterTankMapper, GoalStatus
from GridBasedPlanner import GridBasedPlanner
from GoalsSpawner import GoalsSpawner
from PathSpawner import PathSpawner


def find_and_execute_plan(planner:GridBasedPlanner, path_spawner:PathSpawner, spawned_now:list, executor:PlanExecutor, goal_status:GoalStatus):
    rospy.loginfo("Searching for free-collision shooting trajectories")
    if planner.find_firing_trajectories(ids=spawned_now, goal_status=goal_status, 
                                        exact_collision_check=False, border=1, time_step=0.02, pitch_steps=50, search_retries=10):
        planner.find_driving_plan(sort_by_zones=False, goal_status=goal_status)
        planner.build_plan()

        paths = list(planner.simplified_paths_world.values())
        path_spawner.spawn_paths(paths)

        try:
            executor.execute_queue()
            path_spawner.delete_all()
            rospy.loginfo("Finished current plan")
            return True
        except rospy.ROSInterruptException:
            return False
    else:
        print(f"Nothing to execute for goals status {goal_status}.")
        return False
    

def world_name_callback(msg):
    global world_name
    world_name = msg.data
    

def update_odometry(msg):
        """
        Update the tank's position and orientation from odometry data.
        """
        global tank_position
        tank_position.x = msg.pose.pose.position.x
        tank_position.y = msg.pose.pose.position.y


if __name__ == '__main__':
    rospy.init_node("main_routine")
    rospy.Subscriber('/odom', Odometry, update_odometry)
    rospy.Subscriber('gazebo_world_name', String, world_name_callback)

    tank_position = Pose2D()
    world_name = ""
    # world_name = sys.argv[1] if len(sys.argv) > 1 else "by_small"
    while not world_name:
        pass

    print(f"Loading world: {world_name}")

    mapper = GazeboWaterTankMapper(world_path=f"../worlds/{world_name}.sdf", init=True, 
                                   max_particle_speed=10.0, add_goals=False, goals_dir="../worlds/goals_tmp_for_map")
    print(f"Current tank position: {(tank_position.x, tank_position.y, 0.0)}")

    planner = GridBasedPlanner(mapper,last_tank_position=(tank_position.x, tank_position.y, 0.0))

    goals_spawner = GoalsSpawner(planner)
    path_spawner = PathSpawner()
    mapper.add_goals_from_dir(delete_files=False)
    executor = PlanExecutor(planner, path_spawner)
    goals_group_sizes = [8, 1, 1, 3, 1]

    # Spawn goals in groups and execute plans
    for n in goals_group_sizes:
        spawned_now = goals_spawner.spawn_n_goals(n)
        find_and_execute_plan(planner, path_spawner, spawned_now, executor, goal_status=GoalStatus.QUEUED)

    # Try to reach failed goals
    failed = [goal_id for goal_id, goal in mapper.goals.items() if goal["status"] == GoalStatus.FAILED]
    print(failed)
    find_and_execute_plan(planner, path_spawner, None, executor, goal_status=GoalStatus.FAILED)

    # Try to reach goals marked UNREACHABLE previously
    find_and_execute_plan(planner, path_spawner, None, executor, goal_status=GoalStatus.UNREACHABLE)

    rospy.loginfo("Finished all!")
