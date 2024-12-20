#!/usr/bin/env python3

import rospy
import random
import csv
import os
from time import time
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from geometry_msgs.msg import Pose2D
from PlanExecutor import PlanExecutor
from Mapper import GazeboWaterTankMapper, GoalStatus
from GridBasedPlanner import GridBasedPlanner
from GoalsSpawner import GoalsSpawner
from PathSpawner import PathSpawner


def create_stats_file(filename):
    """
    Create a CSV file with the stats header.
    """
    header = ["simulation", "reached", "failed", "unreachable", "world_name", "simulation_time", "calculations_time"]
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)


def add_stats(filename, row_data):
    """
    Write a row of stats data to the CSV file.
    """
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row_data)


def divide_integer_into_random_parts(total, min_size, max_size, min_components, max_components):
    """
    Divide an integer into a random number of random-sized components that sum up to the total.
    """
    if total < min_components * min_size or total > max_components * max_size:
        raise ValueError("Impossible to divide the total with given constraints.")
    
    num_components = random.randint(min_components, max_components)
    remaining = total
    parts = []
    while remaining > max_size or sum(parts) < total:
        remaining = total
        parts = []
        for _ in range(num_components - 1):
            max_part = min(max_size, remaining - (num_components - len(parts) - 1) * min_size)
            min_part = min_size

            part = random.randint(min_part, max_part)
            parts.append(part)
            remaining -= part

        parts.append(remaining)
    random.shuffle(parts)

    return parts


def init_simulation(generate_goals_sdf=True):
    mapper = GazeboWaterTankMapper(world_path=f"../worlds/{world_name}.sdf", init=True, 
                                   max_particle_speed=10.0, add_goals=False, goals_dir="../worlds/goals_tmp_for_map")
    print(f"Current tank position: {(tank_position.x, tank_position.y, 0.0)}")

    planner = GridBasedPlanner(mapper,last_tank_position=(tank_position.x, tank_position.y, 0.0))

    goals_spawner = GoalsSpawner(planner, generate_sdf=generate_goals_sdf)

    mapper.add_goals_from_dir(delete_files=False)
    executor = PlanExecutor(planner, path_spawner)

    return mapper, planner, executor, goals_spawner


def find_and_execute_plan(planner:GridBasedPlanner, path_spawner:PathSpawner, spawned_now:list, executor:PlanExecutor, goal_status:GoalStatus):
    rospy.loginfo("Searching for free-collision shooting trajectories")
    start = time()
    if planner.find_firing_trajectories(ids=spawned_now, goal_status=goal_status, 
                                        exact_collision_check=False, border=1, time_step=0.02, pitch_steps=50, search_retries=10):
        planner.find_driving_plan(sort_by_zones=False, goal_status=goal_status)
        planner.build_plan()
        calculations_time = time() - start

        paths = list(planner.simplified_paths_world.values())
        path_spawner.spawn_paths(paths)

        try:
            executor.execute_queue()
            path_spawner.delete_all()
            rospy.loginfo("Finished current plan")
            return calculations_time
        except rospy.ROSInterruptException:
            return 0
    else:
        print(f"Nothing to execute for goals status {goal_status}.")
        return 0
    

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
    while not world_name:
        pass

    print(f"Loading world: {world_name}")
    path_spawner = PathSpawner()
    
    # generate goals groups sizes for 10 simulations (total goals 20, 3 to 10 groups, from 1 to 10 goals in each group)
    goals_groups_sizes = [divide_integer_into_random_parts(20, 1, 8, 3, 8) for _ in range(10)]
    rospy.loginfo(f"Generated groups for 10 simulations:\n{goals_groups_sizes}")
    
    stats_file_path = "../stats.csv"
    if not os.path.exists(stats_file_path):
        create_stats_file(stats_file_path)

    for i, goals_group_sizes in enumerate(goals_groups_sizes):
        mapper, planner, executor, goals_spawner = init_simulation()

        reached = 0
        failed = 0
        unreachable = 0
        calculations_time = 0
        start = time()

        for n in goals_group_sizes:
            # Spawn goals in groups and execute plans
            spawned_now = goals_spawner.spawn_n_goals(n)
            calculations_time += find_and_execute_plan(planner, path_spawner, spawned_now, executor, goal_status=GoalStatus.QUEUED)

        # Try to reach failed goals
        calculations_time += find_and_execute_plan(planner, path_spawner, None, executor, goal_status=GoalStatus.FAILED)
        # Try to reach goals marked UNREACHABLE previously
        calculations_time += find_and_execute_plan(planner, path_spawner, None, executor, goal_status=GoalStatus.UNREACHABLE)

        rospy.loginfo(f"Finished simulation {i}.")

        reached += sum(1 for goal in mapper.goals.values() if goal["status"] == GoalStatus.ELIMINATED)
        failed += sum(1 for goal in mapper.goals.values() if goal["status"] == GoalStatus.FAILED)
        unreachable += sum(1 for goal in mapper.goals.values() if goal["status"] == GoalStatus.UNREACHABLE)

        # header = ["simulation", "reached", "failed", "unreachable", "world_name", "simulation_time", "calculations_time"]
        add_stats(stats_file_path, [i, reached, failed, unreachable, world_name, time() - start, calculations_time])

        goals_spawner.delete_all()

    rospy.loginfo("Finished all!")
