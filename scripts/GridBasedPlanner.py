#!/usr/bin/env python3

import heapq
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
from time import time
from itertools import permutations
from queue import Queue
from Mapper import GazeboWaterTankMapper, GoalStatus


class GridBasedPlanner:
    def __init__(self, mapper: GazeboWaterTankMapper, last_tank_position=(0, -9.5, 0)):
        self.mapper = mapper
        self.shooting_data = {}
        self.circles_points = []
        self.paths_world = {}
        self.simplified_paths_world = {}
        self.paths_grid = {}
        self.simplified_paths_grid = {}
        self.last_tank_position = last_tank_position
        self.goals_queue = Queue()
        self.plan = Queue()

    def is_free_cell(self, x, y):
        map_2d = self.mapper.map_2d_buffered
        if x < 0 or x >= map_2d.shape[0] or y < 0 or y >= map_2d.shape[1]:
            return False
        return map_2d[x, y] == 0

    @staticmethod
    def heuristic(a, b):
        # Euclidean
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    def a_star_search(self, start, goal, timeout=10):
        if not self.is_free_cell(start[0], start[1]) or not self.is_free_cell(goal[0], goal[1]):
            return None
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        frontier = []
        heapq.heappush(frontier, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}

        start = time()
        while frontier and time() - start < timeout:
            _, current = heapq.heappop(frontier)
            if current == goal:
                path = []
                node = goal
                while node is not None:
                    path.append(node)
                    node = came_from[node]
                return list(reversed(path))
            for dx, dy in neighbors:
                nx, ny = current[0] + dx, current[1] + dy
                if self.is_free_cell(nx, ny):
                    new_cost = cost_so_far[current] + ((dx * dx + dy * dy) ** 0.5)
                    if (nx, ny) not in cost_so_far or new_cost < cost_so_far[(nx, ny)]:
                        cost_so_far[(nx, ny)] = new_cost
                        priority = new_cost + self.heuristic((nx, ny), goal)
                        heapq.heappush(frontier, (priority, (nx, ny)))
                        came_from[(nx, ny)] = current
        return None

    @staticmethod
    def direction(a, b):
        return b[0] - a[0], b[1] - a[1]

    def simplify_and_smooth_path(self, path):
        """
        Simplifies the path by:
        1. Reducing the path to critical points (start, end, and corners of straight-line sections).
        2. Smoothing stair-like diagonal sections into single straight diagonals, if possible.
        """
        # First, simplify the path to straight-line sections
        simplified_path = self.simplify_path(path)

        # Then, smooth stair-like diagonal sections
        smoothed_path = self.smooth_path(simplified_path)

        return smoothed_path

    def simplify_path(self, path):
        """
        Simplifies the path by retaining only critical points:
        - Start and end points.
        - Points where the path changes a direction (corners).
        """
        if len(path) <= 2:
            return path  # No simplification needed for short paths

        simplified_path = [path[0]]  # Start with the first point

        for i in range(1, len(path) - 1):
            prev = path[i - 1]
            curr = path[i]
            next_ = path[i + 1]

            # Check if the current point is on a straight line (horizontal, vertical, or diagonal)
            if self.check_turning_point(prev, curr, next_):
                simplified_path.append(curr)  # Add the turning point

        simplified_path.append(path[-1])  # Add the last point
        return simplified_path

    @staticmethod
    def check_turning_point(prev, curr, next_):
        return (next_[0] - curr[0]) * (curr[1] - prev[1]) != (curr[0] - prev[0]) * (next_[1] - curr[1])

    def smooth_path(self, path):
        """
        Further simplifies the path by replacing stair-like diagonal segments with single diagonals.
        """
        if len(path) <= 2:
            return path  # No smoothing needed for short paths

        smoothed_path = [path[0]]  # Start with the first point

        i = 0
        while i < len(path) - 1:
            current = path[i]
            j = i + 1

            # Try to find the longest diagonal segment
            while j < len(path):
                if self.line_of_sight(current, path[j]):
                    j += 1
                else:
                    break

            # Add the endpoint of the valid diagonal segment
            smoothed_path.append(path[j - 1])
            i = j - 1  # Continue from the last valid point

        return smoothed_path

    def line_of_sight(self, p1, p2):
        """
        Check if the straight line from p1 to p2 (both (x_idx, y_idx)) is free of obstacles.
        """
        map_2d = self.mapper.map_2d_buffered
        x1, y1 = p1
        x2, y2 = p2
        dist = max(abs(x2 - x1), abs(y2 - y1))  # Use Chebyshev distance
        xs = np.linspace(x1, x2, dist + 1)
        ys = np.linspace(y1, y2, dist + 1)

        for x, y in zip(xs, ys):
            xi = int(round(x))
            yi = int(round(y))
            if xi < 0 or xi >= map_2d.shape[0] or yi < 0 or yi >= map_2d.shape[1]:
                return False  # Out of bounds
            if map_2d[xi, yi] != 0:
                return False  # Collision
        return True

    def visualize_map(self, start, goal, path=None, ax=None, show=True, add_start=True, add_goal=True, legend=True,
                      start_color="darkgreen", goal_color="red", path_color="green", path_label="Path",
                      goal_label="Goal", dash_pattern=None):
        """
        Visualize the map_2d grid along with start, goal, and path.
        - mapper: GazeboWaterTankMapper object with the map inside
        - start: (x_idx, y_idx)
        - goal: (x_idx, y_idx)
        - path: list of (x_idx, y_idx)
        """
        map_2d = self.mapper.map_2d_buffered
        if ax is None:
            fig, ax = self.mapper.visualize_2d_map(map_2d, show=False)

        # Add start and goal
        if add_start:
            ax.scatter(start[0], start[1], c=start_color, s=100, zorder=5, label="Start", marker="o")
        if add_goal:
            ax.scatter(goal[0], goal[1], c=goal_color, s=100, zorder=6, label=goal_label, marker="o")

        # Plot path if given
        if path is not None and len(path) > 0:
            path_x = [p[0] for p in path]
            path_y = [p[1] for p in path]
            # ax.plot(path_x, path_y, path_colors[0], linewidth=2, zorder=4, label=path_label)
            ax.scatter(path_x, path_y, c=path_color, s=30, zorder=4)
            for i in range(len(path) - 1):
                arrow = FancyArrowPatch(path[i], path[i + 1],
                                        arrowstyle="->", mutation_scale=10, zorder=7, linewidth=1,
                                        color=path_color, linestyle=(0, dash_pattern))
                ax.add_patch(arrow)

        ax.set_title("Plan Map Visualization")
        if legend:
            ax.legend()
        if show:
            plt.show()
        else:
            return ax

    def visualize_plan(self, show=False, save=True, goal_status=GoalStatus.QUEUED):
        fig, ax = self.mapper.visualize_2d_map(self.mapper.map_2d_buffered, show=False)
        legend = True
        for goal_id, path in self.simplified_paths_grid.items():
            if self.mapper.goals[goal_id]["status"] == goal_status:
                ax = self.visualize_map(path[0], path[-1], path, ax=ax, show=False, legend=legend,
                                        goal_color="darkblue", path_label="Driving Plan",
                                        goal_label="Driving Goal" if legend else None)

                shooting_goal_coord = self.mapper.world_to_grid(*self.mapper.goals[goal_id]["data"]["position"])[:2]
                ax = self.visualize_map(path[-1], shooting_goal_coord, [path[-1], shooting_goal_coord], ax=ax,
                                        show=False, legend=legend, add_start=False,
                                        goal_color="red", path_color="blue",
                                        path_label="Shooting Plan" if legend else None,
                                        goal_label="Shooting Goal", dash_pattern=(2, 2))
                legend = False

        if show:
            plt.show()
        if save:
            plt.savefig("grid_based_driving_plan.png", dpi=300, bbox_inches="tight")

    def find_firing_trajectories(self, distance_between_points=0.25, radius_degradation=0.5, min_radius=0.5, ids=None,
                                 reset=True, goal_status=GoalStatus.QUEUED, search_retries=5,
                                 exact_collision_check=False, border=0, time_step=0.01, pitch_steps=200):
        start = time()
        if reset:
            self.reset_plans()

        goals = self.mapper.goals
        goals_ids = ids if ids is not None else list(goals.keys())

        centers = [(goal_id, (*goals[goal_id]["data"]["position"][:2],
                              goals[goal_id]["data"]["position"][2] + goals[goal_id]["data"]["size"][2] / 2))
                   for goal_id in goals_ids if goals[goal_id]["status"] == goal_status]
        centers = sorted(centers, key=lambda c: np.linalg.norm(np.array(c[1]) - np.array(self.last_tank_position)))

        tank_position_grid = self.mapper.world_to_grid(*self.last_tank_position)

        for goal_id, (x, y, z) in centers:
            trajectory_found = False
            radius = self.mapper.calculate_max_height_and_horizontal_distance(0.0, z)[1]
            circle_points = []

            while radius >= min_radius:
                n_samples = int(2 * np.pi * radius / distance_between_points)
                circle_points += self.mapper.generate_circle_samples((x, y), radius, n_samples)
                radius -= radius_degradation
            # sort potential shooting source points for the current target (circle center), relatively to the distance from the tank absolute gun position (ascending)
            circle_points = sorted(circle_points,
                                   key=lambda p: np.linalg.norm(np.array(p) - np.array(self.last_tank_position)))
            circle_points = sorted(circle_points,
                                   key=lambda p: self.traj_sort_key(p[:2], self.last_tank_position[:2], (x, y),
                                                                    (0.9, 0.1)))
            self.circles_points.append(circle_points)

            retries = 0
            for point in circle_points:
                trajectory, pitch, yaw, v0, t = self.mapper.calculate_particle_trajectory(gun_base=point,
                                                                                          target=(x, y, z),
                                                                                          exact=exact_collision_check,
                                                                                          border=border,
                                                                                          time_step=time_step,
                                                                                          pitch_steps=pitch_steps)
                if trajectory is not None:
                    if self.a_star_search(tank_position_grid[:2], self.mapper.world_to_grid(*point)[:2]) is not None:
                        print(f"Trajectory found for {x, y, z}!")
                        self.shooting_data[goal_id] = {"trajectory": trajectory, "pitch": pitch, "yaw": yaw, "v0": v0,
                                                       "t": t, "driving_goal": point}
                        trajectory_found = True
                        break
                    else:
                        retries += 1
                        if retries >= search_retries:
                            print(f"Reachable trajectory not found for {x, y, z} within {retries} tries.")
                            break
            if not trajectory_found:
                self.mapper.goals[goal_id]["status"] = GoalStatus.UNREACHABLE
                print(f"Trajectory for {x, y, z} doesn't exist. No solution.")

        print(f"Took:{time() - start}")
        if len(self.shooting_data) > 0:
            return True
        return False

    @staticmethod
    def traj_sort_key(shooting_point, start, target, weigths=(1, 1)):
        """
        Key function to calculate the sorting metric for a point.
        The metric is the sum of:
        1. Distance from the point to the given point.
        2. Perpendicular distance from the point to the line connecting the given point and the line point.
        Parameters:
            shooting_point (tuple): The point (x, y) to evaluate.
            start (tuple): The given point (x, y).
            target (tuple): A point that defines the line passing through the given point.
        Returns:
            float: The sorting metric for the point.
        """

        def distance_to_point(p1, p2):
            """Calculate Euclidean distance between two points."""
            return np.linalg.norm(np.array(p1) - np.array(p2))

        def perpendicular_distance(point, line_start, line_end):
            """Calculate perpendicular distance from a point to a line defined by two points."""
            x0, y0 = point
            x1, y1 = line_start
            x2, y2 = line_end
            # Line equation: ax + by + c = 0
            a = y2 - y1
            b = x1 - x2
            c = x2 * y1 - x1 * y2
            # Perpendicular distance formula
            return abs(a * x0 + b * y0 + c) / np.sqrt(a ** 2 + b ** 2)

        # Combine the two metrics
        return weigths[0] * distance_to_point(shooting_point, start) + weigths[1] * perpendicular_distance(
            shooting_point, start, target)

    def find_driving_plan(self, sort_by_zones=True, goal_status=GoalStatus.QUEUED):
        drive_goals = self.sorted_goals(by_zone=sort_by_zones, goal_status=goal_status)
        start_grid = self.mapper.world_to_grid(*self.last_tank_position)[:2]  # tank_position in grid coordinates
        drive_goals_grid = [(goal_id, self.mapper.world_to_grid(*goal)[:2]) for goal_id, goal in
                            drive_goals]  # goals in grid coordinates

        current_tank_position = start_grid
        for i, (goal_id, goal) in enumerate(drive_goals_grid):
            path = self.a_star_search(current_tank_position, goal)
            if path is not None:
                print(f"Path found for goal {drive_goals[i]}!")
                # Simplify the path
                word_path = [self.mapper.grid_to_world(x, y) for x, y in path]
                simplified_path = self.simplify_and_smooth_path(path)
                simplified_world_path = [self.mapper.grid_to_world(x, y) for x, y in simplified_path]
                self.paths_grid[goal_id] = path
                self.paths_world[goal_id] = word_path
                self.simplified_paths_grid[goal_id] = simplified_path
                self.simplified_paths_world[goal_id] = simplified_world_path
                current_tank_position = goal

                self.goals_queue.put(goal_id)
                # self.mapper.goals[goal_id]["status"] = GoalStatus.QUEUED
            else:
                print(f"No path found for goal {drive_goals[i]} within reasonable time.")

        self.last_tank_position = self.mapper.grid_to_world(*current_tank_position)

    def sorted_goals(self, k=6, by_zone=True, goal_status=GoalStatus.QUEUED):
        # get new goals
        drive_goals = [(goal_id, data["driving_goal"]) for goal_id, data in self.shooting_data.items() if
                       self.mapper.goals[goal_id]["status"] == goal_status]

        if by_zone:
            x_min, x_max = min(drive_goals, key=lambda point: point[1][0])[1][0], \
            max(drive_goals, key=lambda point: point[1][0])[1][0]
            y_min, y_max = min(drive_goals, key=lambda point: point[1][1])[1][1], \
            max(drive_goals, key=lambda point: point[1][1])[1][1]
            x_center = (x_max + x_min) / 2
            y_center = (y_max + y_min) / 2
            # zones order: upper-left, upper-right, down-right, down-left
            # local order: x_min, x_max, y_min, y_max
            zones = [(x_min, x_center, y_center, y_max),
                     (x_center, x_max, y_center, y_max),
                     (x_center, x_max, y_min, y_center),
                     (x_min, x_center, y_min, y_center)]
            # divide all goals by zones on the map (currently 4 rectangles)
            goals_by_zone = [[] for _ in range(len(zones))]
            for goal in drive_goals:
                for i, (x_min, x_max, y_min, y_max) in enumerate(zones):
                    if x_min <= goal[1][0] <= x_max and y_min <= goal[1][1] <= y_max:
                        goals_by_zone[i].append(goal)
                        break
            # Sort zones by the distance of the centroids from the current tank position
            goals_by_zone = [[tuple(np.mean([goal[1] for goal in goals], axis=0)), goals] for goals in goals_by_zone if
                             len(goals) > 0]  # add centroids for each non-empty zone
            optimal_centroids_order = self.brute_force_tsp([(i, goals[0]) for i, goals in enumerate(goals_by_zone)])
            order_map = {c[1]: i for i, c in enumerate(optimal_centroids_order)}
            goals_by_zone = sorted(goals_by_zone, key=lambda goals: order_map[goals[0]])

            # sort goals in each zone by distance of centroid from the current tank position (ascending), then concatenate
            for i in range(len(goals_by_zone)):
                goals_by_zone[i][1] = sorted(goals_by_zone[i][1],
                                             key=lambda point: np.linalg.norm(np.array(point[1]) - goals_by_zone[i][0]))
            drive_goals = sum([goals[1] for goals in goals_by_zone], [])
        else:
            drive_goals = sorted(drive_goals,
                                 key=lambda goal: np.linalg.norm(goal[1] - np.array(self.last_tank_position)))

        # split tank starting position + goals into groups of k elements (at most) and find an optimal path for each group
        drive_goals = [([("start", self.last_tank_position)] + drive_goals)[i:i + k] for i in
                       range(0, len(drive_goals) + 1, k)]
        drive_goals_sorted = []
        # using tsp brute force solver (for small number of points) for an optimal path in the group
        for i, group in enumerate(drive_goals):
            if drive_goals_sorted:
                group = drive_goals_sorted[-1:] + group
            optimal_path = self.brute_force_tsp(group)
            drive_goals_sorted += optimal_path[1:]

        return drive_goals_sorted

    @staticmethod
    def compute_distance(points, path):
        """
        Compute the total distance of the given path.
        Parameters:
            points (list of tuples): List of (x, y) coordinates for each point.
            path (list): A sequence of indices representing the path.
        Returns:
            float: Total distance of the path.
        """
        distance = 0
        for i in range(len(path) - 1):
            distance += np.linalg.norm(np.array(points[path[i]][1]) - np.array(points[path[i + 1]][1]))
        return distance

    def brute_force_tsp(self, points, k=6):
        """
        Solve the Traveling Salesman Problem (TSP) using brute-force approach. Assuming here no more than 6 goals
        Parameters:
            points (list of tuples): List of (x, y) coordinates for each point.
            k (int): maximum points allowed for the method to be efficient.
        Returns:
            tuple: Optimal path and its total distance.

        """
        if len(points) > k + 1:
            raise ValueError(f"This method is not efficient for more than {k + 1} goals.")

        goal_indices = list(range(1, len(points)))  # Exclude start point (index 0)
        all_permutations = permutations(goal_indices)

        optimal_path = None
        min_distance = float("inf")

        for perm in all_permutations:
            path = [0] + list(perm)
            distance = self.compute_distance(points, path)
            if distance < min_distance:
                min_distance = distance
                optimal_path = path

        return [points[i] for i in optimal_path]

    def build_plan(self):
        # commands = ["drive", "aim", "shoot"]
        while not self.goals_queue.empty():
            goal_id = self.goals_queue.get()

            path = self.simplified_paths_world[goal_id]
            for coord in path[
                         1:]:  # skip the first coordinate because it is the tank position at the start of each path
                self.add_command("drive", goal_id=goal_id, x=coord[0], y=coord[1], theta=0)

            shoot_data = self.shooting_data[goal_id]
            self.add_command("aim", pitch=shoot_data["pitch"], yaw=shoot_data["yaw"], relative=0)
            self.add_command("shoot", goal_id=goal_id, v0=shoot_data["v0"], t=shoot_data["t"],
                             pitch=shoot_data["pitch"],
                             goal_position=self.mapper.goals[goal_id]["data"]["position"])
            self.add_command("aim", pitch=0.0, yaw=0.0, relative=1)

    def add_command(self, command_type, **kwargs):
        command = {'type': command_type, 'params': kwargs}
        self.plan.put(command)

    def reset_plans(self):
        self.circles_points = []
        self.shooting_data = {}
        self.paths_grid = {}
        self.paths_world = {}
        self.simplified_paths_grid = {}
        self.simplified_paths_world = {}
        self.goals_queue = Queue()
        self.plan = Queue()
