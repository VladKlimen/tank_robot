import xml.etree.ElementTree as ET
import os
from itertools import product
from enum import Enum, auto
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba, ListedColormap
from PIL import Image
from scipy.ndimage import binary_dilation
from scipy.spatial.transform import Rotation as R


def is_running_in_jupyter():
    try:
        # Check if 'get_ipython' exists and is not None
        from IPython import get_ipython
        if get_ipython() is not None:
            return True
    except ImportError:
        pass
    return False

if is_running_in_jupyter():
    import pyvista as pv
    pv.set_jupyter_backend("trame")

# Goals statuses: "new", "queued", "executing", "unreachable", "eliminated", "failed"
class GoalStatus(Enum):
    NEW = auto()
    QUEUED = auto()
    EXECUTING = auto()
    UNREACHABLE = auto()
    ELIMINATED = auto()
    FAILED = auto()


class GazeboWaterTankMapper:
    def __init__(self, world_path, goals_dir=None, init=True, grid_half_length=10, gun_length=0.25, tank_height=0.2, resolution=0.04,
                 bounds=None, round_to=4, circle_nsamples=50, goal_sizes=None, add_floor=True, add_goals=True, floor_ext=0,
                 max_particle_speed=10.0, min_gun_angle=-np.pi / 6, max_gun_angle=np.pi / 4,
                 gun_offset=(0.255, 0.0,  0.042), gun_joint_offset=(0.0, 0.0, 0.11)):
        """
        Parameters:
            file_path (str): The path to the Gazebo world file.
            circle_nsamples (int): for a cylinder object, number of points to sample around the circle
            resolution (float): The size of each grid cell.
        """
        self.file_path = world_path
        self.goals_dir = goals_dir
        self.round_to = round_to
        self.circle_nsamples = circle_nsamples
        self.add_floor = add_floor
        self.add_goals = add_goals
        self.resolution = resolution
        self.floor_ext = floor_ext
        self.max_particle_speed = max_particle_speed
        self.min_gun_angle = min_gun_angle
        self.max_gun_angle = max_gun_angle
        self.gun_offset = gun_offset
        self.gun_joint_offset = gun_joint_offset

        self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max = (
                [-grid_half_length, grid_half_length] * 2 + [0.0, grid_half_length]) if bounds is None else bounds

        self.nlayers_2d = int(tank_height / resolution + 1)
        self.gun_radius = int(gun_length / resolution + 1)

        self.goal_sizes = {"goal_1": [0.5, 0.205, 0.372],
                           "goal_2": [0.46, 0.36, 0.178],
                           "goal_3": [1.275, 0.343, 1.315],
                           "goal_4": [0.322, 0.205, 0.464],
                           } if goal_sizes is None else goal_sizes

        self.codes = {"empty": 0, "floor": 10, "box": 255, "cylinder": 255, "goal": 5, "point": 6, "buffer": 7,
                      "circle": 8, "path": 9, "node": 11}
        self.allowed_to_collide = [self.codes["empty"], self.codes["point"], self.codes["buffer"], self.codes["circle"],
                                   self.codes["path"], self.codes["node"]]
        self.active_codes = {"empty": False, "floor": False, "box": False, "cylinder": False, "goal": False,
                             "point": False, "buffer": False, "circle": False, "path": False, "node": False}
        self.colors = {0: "#FFFFFF", 10: "#FFF3F0", 255: "#10606A", 5: "#AD2327", 6: "#424BAA", 7: "#F76D6C",
                       8: "#DA23FF", 9: "#2AFF00", 11: "#FF8600"}
        
        self.goals_codes = {"last_serial": 1000, "codes": {}}

        self.grid = None
        self.map_2d = None
        self.map_2d_buffered = None
        self.goals = {}
        self.obstacles, self.others = [], []

        if init is True:
            print("Initializing map")
            self.parse_gazebo_world()
            self.calculate_z_max()
            self.create_3d_grid()
            if self.add_floor:
                self.add_floor_to_grid(floor_ext)
            self.add_all_to_grid(add_goals=self.add_goals)
            self.create_2d_map_from_layers()
            self.add_buffer_zone(buffer_length=self.gun_radius)
            print("Initialized.")

    def parse_gazebo_world(self):
        """
        Parse a Gazebo .world or .sdf file to extract obstacle data.
        """
        print("Parsing Gazebo World file...", end="")
        tree = ET.parse(self.file_path)
        root = tree.getroot()

        # Parse for models (assuming obstacles/goals are represented as models in the .world/.sdf file)
        for model in root.findall(".//model"):
            name = model.get("name")
            if "_obs" in name.lower():
                obstacle = self.get_obstacle_data(model)
                self.obstacles.append(obstacle)
            elif "_goal" in name.lower():
                goal_id, goal = self.get_goal_data(model)
                self.goals[goal_id] = {"data": goal, "status": GoalStatus.NEW}
                self.goals_codes["codes"][goal_id] = self.codes["goal"] + self.goals_codes["last_serial"]
                self.goals_codes["last_serial"] += 1
            else:
                self.others.append(model.get("name"))
        print("\tV")

    def add_goals_from_dir(self, delete_files=True):
        """
        Add goals from sdf files in goals_dir
        """
        if not self.goals_dir:
            print("Goals directory path wasn't provided.")
            return False
        
        print("Adding goals from directory...", end="")
        for file_name in self.get_file_names(self.goals_dir):
            file_path = os.path.join(self.goals_dir, file_name)
            tree = ET.parse(file_path)
            root = tree.getroot()

            # Parse for models (assuming goals are represented as models in the .world/.sdf file)
            for model in root.findall(".//model"):
                name = model.get("name")
                if "_goal" in name.lower():
                    goal_id, goal = self.get_goal_data(model)
                    self.goals[goal_id] = {"data": goal, "status": GoalStatus.NEW}
                    self.goals_codes["codes"][goal_id] = self.codes["goal"] + self.goals_codes["last_serial"]
                    self.goals_codes["last_serial"] += 1
            if delete_files:
                self.delete_file(file_path)
        print("\tV")

    @staticmethod
    def get_file_names(directory_path):
        # Get files from the given dir path
        file_names = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
        return file_names
    
    @staticmethod
    def delete_file(file_path):
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
            else:
                print(f"File '{file_path}' does not exist.")
        except Exception as e:
            print(f"Error while deleting file: {e}")

    def add_all_to_grid(self, add_goals=True):
        print("Adding all world elements to the grid", end="")
        boxes = sum([[elem for elem in list(obs.values()) if elem["type"] == "box"] for obs in self.obstacles], [])
        cylinders = sum([[elem for elem in list(obs.values()) if elem["type"] == "cylinder"] for obs in self.obstacles], [])
        for box in boxes:
            self.add_box_to_grid(box)
        for cylinder in cylinders:
            self.add_cylinder_to_grid(cylinder)
        if add_goals:
            for goal in self.goals.values():
                self.add_box_to_grid(goal["data"], box_type="goal")
            if len(self.goals) > 0:
                self.active_codes["goal"] = True

        if len(boxes) > 0:
            self.active_codes["box"] = True
        if len(cylinders) > 0:
            self.active_codes["cylinder"] = True

        print("\tV")

    def calculate_z_max(self):
        print("Calculating z max...", end="")
        self.z_max = max(self.get_max_height(),
                         self.calculate_max_height_and_horizontal_distance()[0]) + self.resolution
        print("\tV")

    def get_obstacle_data(self, model):
        """
        Extract geometry and pose data for all obstacle components (links collisions)
        within a given model element from the SDF/Gazebo XML structure.

        Parameters:
            model (xml.etree.ElementTree.Element): The XML element representing a model.

        Returns:
            dict: A dictionary mapping each component to its geometry (box or cylinder)
                  and absolute pose. If no geometry is found, the dictionary may be empty.
        """
        model_pose = self.get_pose(model)
        data = {}
        for i, link in enumerate(model.findall(".//link")):
            link_pose = self.get_pose(link)

            for j, collision in enumerate(link.findall(".//collision")):
                collision_pose = self.get_pose(collision)
                geometry = collision.find("geometry")
                if geometry is not None:
                    box = self.get_box(geometry, collision_pose, link_pose, model_pose)
                    if box is not None:
                        data[f"obs_{i}{j}"] = box
                        continue

                    cylinder = self.get_cylinder(geometry, collision_pose, link_pose, model_pose)
                    if cylinder is not None:
                        data[f"obs_{i}{j}"] = cylinder
                        continue
        return data

    def get_goal_data(self, model):
        size = self.get_goal_size(model.get("name"))
        pose = self.get_pose(model)
        goal_id = model.get("name")
        # Assuming the scale is present in the model (no additional checks)
        scale = model.find("link").find("visual").find("geometry").find("mesh").find("scale").text.strip().split()
        scale = list(map(float, scale))
        goal = {"type": "goal",
                "size": [round(a * b, self.round_to) for a, b in zip(size, scale)],
                "position": tuple(pose[:3]),
                "orientation": tuple(pose[3:])
                }
        return goal_id, goal

    def get_goal_size(self, name):
        # The goal sizes are hardcoded, according to the mesh data of the goals (instead of complex mesh consider goals as boxes)
        for key in self.goal_sizes:
            if key in name:
                return self.goal_sizes[key]
        return None

    def get_pose(self, elem):
        """
        Retrieve the pose (position and orientation) from an SDF element that may contain a <pose> tag.

        Parameters:
            elem (xml.etree.ElementTree.Element): The XML element possibly containing a <pose> element.

        Returns:
            list of float: A 6-element list [x, y, z, roll, pitch, yaw], representing the pose.
                           If no <pose> is found, returns [0.0]*6 by default.
        """
        pose = elem.find("pose")
        if pose is not None:
            pose = [round(p, self.round_to) for p in list(map(float, pose.text.strip().split()))]
        else:
            pose = [0.0] * 6
        return pose

    def get_box(self, geometry, collision_pose, link_pose, model_pose):
        """
        Extract box geometry data and compute its absolute position/orientation.

        Parameters:
            geometry (xml.etree.ElementTree.Element): The "geometry" element containing a "box".
            collision_pose (list of float): The [x, y, z, roll, pitch, yaw] pose of the collision element.
            link_pose (list of float): The [x, y, z, roll, pitch, yaw] pose of the link.
            model_pose (list of float): The [x, y, z, roll, pitch, yaw] pose of the model.

        Returns:
            dict or None: If a box is found, returns a dictionary with keys. Otherwise, returns None if no box is found.
        """
        box = geometry.find("box")
        if box is not None:
            size = list(map(float, box.find("size").text.strip().split()))
            position, orientation = self.get_abs_pose(model_pose, sum(self.get_abs_pose(link_pose, collision_pose), []))
            box = {"type": "box",
                   "size": size,
                   "position": position,
                   "orientation": orientation
                   }
        return box

    def get_cylinder(self, geometry, collision_pose, link_pose, model_pose):
        """
        Extract cylinder geometry data and compute its absolute position/orientation.

        Parameters:
            geometry (xml.etree.ElementTree.Element): The "geometry" element containing a "cylinder".
            collision_pose (list of float): The [x, y, z, roll, pitch, yaw] pose of the collision element.
            link_pose (list of float): The [x, y, z, roll, pitch, yaw] pose of the link.
            model_pose (list of float): The [x, y, z, roll, pitch, yaw] pose of the model.

        Returns:
            dict or None: If a cylinder is found, returns a dictionary with keys. Otherwise, returns None if no cylinder is found.
        """
        cylinder = geometry.find("cylinder")
        if cylinder is not None:
            radius = float(cylinder.find("radius").text.strip())
            length = float(cylinder.find("length").text.strip())
            position, orientation = self.get_abs_pose(model_pose, sum(self.get_abs_pose(link_pose, collision_pose), []))
            cylinder = {"type": "cylinder",
                        "radius": radius,
                        "length": length,
                        "position": position,
                        "orientation": orientation
                        }
        return cylinder

    def get_abs_pose(self, model_pose, link_pose):
        """
        Compute the absolute pose of a link/collision by combining its pose with the model"s pose.

        Parameters:
            model_pose (list of float): The model pose [x, y, z, roll, pitch, yaw].
            link_pose (list of float): The link or collision pose [x, y, z, roll, pitch, yaw].

        Returns:
            tuple: (pos_link_abs, rot_link_abs) where:
                   pos_link_abs: [px, py, pz] absolute position
                   rot_link_abs: [roll, pitch, yaw] absolute orientation (in radians)
        """
        if all(x == 0 for x in link_pose):
            return model_pose[:3], model_pose[3:]
        if all(x == 0 for x in model_pose):
            return link_pose[:3], link_pose[3:]

        rot_model = R.from_euler("xyz", model_pose[3:])
        rot_link = R.from_euler("xyz", link_pose[3:])
        rot_link_abs = [round(r, self.round_to) for r in list((rot_model * rot_link).as_euler("xyz"))]

        model_pos, link_pos = model_pose[:3], link_pose[:3]

        pos_link_abs = [round(p, self.round_to) for p in
                        list(rot_model.apply(np.array(link_pos)) + np.array(model_pos))]

        return pos_link_abs, rot_link_abs

    def compute_absolute_height(self, obj):
        """
        Compute the maximum global Z-coordinate of the object top.

        Parameters:
            obj (dict): Dictionary describing the object with keys:
                        - "type": "box" or "cylinder"
                        - "size": For box [sx, sy, sz], for cylinder [radius, length]
                        - "position": [px, py, pz]
                        - "orientation": [roll, pitch, yaw] in radians
        Returns:
            float: The maximum global Z-coordinate of the top surface of the object.
        """
        obj_type = obj["type"]
        size = obj["size"] if obj_type == "box" else (obj["radius"], obj["length"])
        position = np.array(obj["position"])
        orientation = np.array(obj["orientation"])

        # Create a rotation object from the given Euler angles
        # Assuming "xyz" order: roll about x, pitch about y, yaw about z
        rot = R.from_euler("xyz", orientation)

        if obj_type == "box":
            sx, sy, sz = size
            hx, hy, hz = sx / 2, sy / 2, sz / 2  # Half-dimensions (objects are centered across their size)
            # Top face corners in local coordinates at z = +hz
            corners_local = np.array(np.meshgrid([-hx, hx], [-hy, hy], [-hz, hz])).T.reshape(-1, 3)
            # Rotate and translate corners
            corners_world = position + rot.apply(corners_local)
            # Find the maximum z-value among top corners
            max_z = np.max(corners_world[:, 2])
            return max_z

        elif obj_type == "cylinder":
            radius, length = size
            hz = length / 2.0
            # The top face is a circle at local z = +hz
            angles = np.linspace(0, 2 * np.pi, self.circle_nsamples, endpoint=False)
            # Local points on the top face circle
            # x = r*cos(theta), y = r*sin(theta), z = hz
            circle_points_local = np.column_stack([
                radius * np.cos(angles),
                radius * np.sin(angles),
                np.full(self.circle_nsamples, hz)
            ])
            # Rotate and translate
            circle_points_world = position + rot.apply(circle_points_local)
            # Max z on the top circle
            max_z = np.max(circle_points_world[:, 2])
            return max_z
        else:
            raise ValueError("Unsupported object type. Must be box or cylinder.")

    def get_max_height(self):
        """
        Determine the highest global Z-coordinate among a list of obstacles.

        Returns:
            float: The maximum top height (Z-coordinate) found among all provided obstacles.
        """
        heights = []
        for obs in self.obstacles:
            for obj in obs.values():
                heights.append(self.compute_absolute_height(obj))
        return max(heights)

    def create_3d_grid(self):
        """
        Create a 3D occupancy grid with specified bounds and resolution.

        Returns:
            tuple: (grid, x_range, y_range, z_range)
                   grid: 3D numpy array initialized to zeros representing empty cells.
                   x_range, y_range, z_range: The coordinate arrays defining the grid axes.
        """
        print("Creating 3D grid...", end="")
        x_range = np.arange(self.x_min, self.x_max, self.resolution)
        y_range = np.arange(self.y_min, self.y_max, self.resolution)
        z_range = np.arange(self.z_min, self.z_max, self.resolution)
        grid_shape = (len(x_range), len(y_range), len(z_range))
        self.grid = np.zeros(grid_shape, dtype=np.uint8)  # Using uint8 to save memory
        print("\tV")

    def world_to_grid(self, x, y, z=None):
        """
        Convert world coordinates to 3D grid indices.

        Parameters:
        :params x, y, z (float): The world coordinates of a point.

        Returns:
            tuple of int: (i, j, k) indices in the grid array corresponding to the given world point.
        """
        i = int((x - self.x_min) / self.resolution)
        j = int((y - self.y_min) / self.resolution)
        k = 0 if not self.add_floor else 1
        if z is not None:
            k = int((z - self.z_min) / self.resolution)
        return i, j, k

    def grid_to_world(self, i, j, k=None):
        """
        Convert 3D grid indices to world coordinates.

        Parameters:
        :params i, j, k (int): 3D grid indices corresponding to the world point.

        Returns:
            tuple of float: (z, y, z) - point world approximated coordinates.
        """
        x = round(i * self.resolution + self.x_min, self.round_to)
        y = round(j * self.resolution + self.y_min, self.round_to)
        if k is not None:
            z = round(k * self.resolution + self.z_min, self.round_to)
            return x, y, z
        return x, y, 0

    def add_box_to_grid(self, obstacle, box_type="box", code=None):
        """
        Fill the 3D occupancy grid cells that correspond to the volume occupied by a rotated box.

        Parameters:
            obstacle (dict): A dictionary describing a box, including "size", "position", and "orientation".

        Returns:
            None: The function modifies "grid" in-place, setting grid cells inside the box volume to 1.
        """
        pos = obstacle["position"]
        size = obstacle["size"]
        orientation = obstacle["orientation"]
        rotation = R.from_euler("xyz", orientation)

        # Compute the 8 corners of the box in local coordinates
        l, w, h = size[0] / 2, size[1] / 2, size[2] / 2
        corners = np.array(np.meshgrid([-l, l], [-w, w], [-h, h])).T.reshape(-1, 3)

        # Rotate the corners to world coordinates
        rotated_corners = rotation.apply(corners)
        world_corners = rotated_corners + pos

        # Compute the coordinates of axis-aligned bounding box (AABB)
        x_coords = world_corners[:, 0]
        y_coords = world_corners[:, 1]
        z_coords = world_corners[:, 2]

        # Convert the AABB to grid indices
        i_start, j_start, k_start = self.world_to_grid(x_coords.min(), y_coords.min(), z_coords.min())
        i_end, j_end, k_end = self.world_to_grid(x_coords.max(), y_coords.max(), z_coords.max())

        points_to_add = []
        # Iterate over grid cells within the AABB (clamped indices to grid bounds)
        for i in range(max(i_start, 0), min(i_end, self.grid.shape[0] - 1) + 1):
            for j in range(max(j_start, 0), min(j_end, self.grid.shape[1] - 1) + 1):
                for k in range(max(k_start, int(self.add_floor)), min(k_end, self.grid.shape[2] - 1) + 1):
                    # Compute world coordinates of the grid cell center
                    x = self.x_min + (i + 0.5) * self.resolution
                    y = self.y_min + (j + 0.5) * self.resolution
                    z = self.z_min + (k + 0.5) * self.resolution
                    # Transform to local coordinates
                    point_world = np.array([x, y, z])
                    point_local = rotation.apply(point_world - pos, inverse=True)
                    # Check if point is inside the unrotated box
                    if -l <= point_local[0] <= l and -w <= point_local[1] <= w and -h <= point_local[2] <= h:
                        if self.grid[i, j, k] not in self.allowed_to_collide:
                            if self.grid[i, j, k] != self.codes["box"] and self.grid[i, j, k] != self.codes["cylinder"]:
                                print(f"Failed to add {box_type} at {pos}: collides with other non-obstacle object (code: {self.grid[i, j, k]}).")
                                return False
                            else:
                                continue
                        if self.grid[i, j, k] == self.codes["empty"]:
                            points_to_add.append((i, j, k))
        for i, j, k in points_to_add:
            self.grid[i, j, k] = self.codes[box_type] if not code else code

        return True

    def add_cylinder_to_grid(self, obstacle):
        """
        Fill the 3D occupancy grid cells that correspond to the volume occupied by a rotated cylinder.

        Parameters:
            obstacle (dict): A dictionary describing a cylinder, including "radius", "length",
                             "position", and "orientation".

        Returns:
            None: The function modifies "grid" in-place, setting grid cells inside the cylinder volume to 1.
        """
        pos = obstacle["position"]
        radius = obstacle["radius"]
        length = obstacle["length"]
        orientation = obstacle["orientation"]
        rotation = R.from_euler("xyz", orientation)

        # Compute the endpoints of the cylinder in local coordinates, then rotate and translate endpoints to world coordinates
        h = length / 2
        endpoints = np.array([[0, 0, -h], [0, 0, h]])
        rotated_endpoints = rotation.apply(endpoints) + pos

        # Sample points on the cylinder surface to estimate AABB
        theta = np.linspace(0, 2 * np.pi, self.circle_nsamples)
        z = np.array([-h, h])
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_cyl = radius * np.cos(theta_grid)
        y_cyl = radius * np.sin(theta_grid)
        z_cyl = z_grid

        # Flatten and stack coordinates, then rotate and translate to world coordinates
        points_cylinder_surface = np.vstack((x_cyl.flatten(), y_cyl.flatten(), z_cyl.flatten())).T
        points_world = rotation.apply(points_cylinder_surface) + pos

        # Compute the coordinates of axis-aligned bounding box (AABB)
        x_coords = points_world[:, 0]
        y_coords = points_world[:, 1]
        z_coords = points_world[:, 2]

        # Convert AABB to grid indices
        i_start, j_start, k_start = self.world_to_grid(x_coords.min(), y_coords.min(), z_coords.min())
        i_end, j_end, k_end = self.world_to_grid(x_coords.max(), y_coords.max(), z_coords.max())
        points_to_add = []
        # Iterate over grid cells within the AABB (with clamped indices)
        for i in range(max(i_start, 0), min(i_end, self.grid.shape[0] - 1) + 1):
            for j in range(max(j_start, 0), min(j_end, self.grid.shape[1] - 1) + 1):
                for k in range(max(k_start, int(self.add_floor)), min(k_end, self.grid.shape[2] - 1) + 1):
                    # Compute world coordinates of the grid cell center
                    x = self.x_min + (i + 0.5) * self.resolution
                    y = self.y_min + (j + 0.5) * self.resolution
                    z = self.z_min + (k + 0.5) * self.resolution
                    # Transform to local coordinates
                    point_world = np.array([x, y, z])
                    point_local = rotation.apply(point_world - pos, inverse=True)
                    # Check if point is inside the unrotated cylinder
                    px, py, pz = point_local
                    if px ** 2 + py ** 2 <= radius ** 2 and -h <= pz <= h:
                        if self.grid[i, j, k] not in self.allowed_to_collide:
                            if self.grid[i, j, k] != self.codes["box"] and self.grid[i, j, k] != self.codes["cylinder"]:
                                print(f"Failed to add cylinder: collides with other non-obstacle object (code: {self.grid[i, j, k]}).")
                                return False
                            else:
                                continue
                        if self.grid[i, j, k] == self.codes["empty"]:
                            points_to_add.append((i, j, k))
        for i, j, k in points_to_add:
            self.grid[i, j, k] = self.codes["cylinder"]

        return True
                       

    def add_floor_to_grid(self, extension=1):
        """
        Add a floor to the existing occupancy grid by extending the bounds and adding
        an occupied layer at z = z_min - resolution. The floor extends beyond the original x, y
        bounds by "v" meters in all directions, and z_max remains unchanged.

        Parameters:
            extension (float): The extension value for x and y bounds for the floor.
        """
        print("Adding floor to grid...", end="")
        # Compute new boundaries
        new_x_min = self.x_min - extension
        new_x_max = self.x_max + extension
        new_y_min = self.y_min - extension
        new_y_max = self.y_max + extension
        new_z_min = self.z_min - self.resolution  # the floor is one cell surface below the old z_min
        # Create new coordinate arrays
        new_x_range = np.arange(new_x_min, new_x_max, self.resolution)
        new_y_range = np.arange(new_y_min, new_y_max, self.resolution)
        new_z_range = np.arange(new_z_min, self.z_max, self.resolution)

        new_shape = (len(new_x_range), len(new_y_range), len(new_z_range))
        new_grid = np.zeros(new_shape, dtype=np.uint8)

        # Compute index offsets for placing old grid inside the new one
        # The old grid (i,j,k) was defined from x_min, y_min, z_min. Now we have shifted mins.
        i_offset = int((self.x_min - new_x_min) / self.resolution)
        j_offset = int((self.y_min - new_y_min) / self.resolution)
        k_offset = int((self.z_min - new_z_min) / self.resolution)

        # Insert the old grid into the new grid at the appropriate offset
        old_i_max = i_offset + self.grid.shape[0]
        old_j_max = j_offset + self.grid.shape[1]
        old_k_max = k_offset + self.grid.shape[2]

        new_grid[i_offset:old_i_max, j_offset:old_j_max, k_offset:old_k_max] = self.grid

        # Fill the floor layer (the lowest layer in z)
        # Since new_z_min < z_min, the floor corresponds to k=0 in the new grid coordinate system.
        new_grid[:, :, 0] = self.codes["floor"]

        self.grid, self.x_min, self.y_min, self.z_min, self.x_max, self.y_max = (
            new_grid, new_x_min, new_y_min, new_z_min, new_x_max, new_y_max)

        self.active_codes["floor"] = True
        self.add_floor = True
        print("\tV")

    def add_buffer_to_grid(self, buffer):
        if not self.add_floor:
            print("Buffer wasn't added to the 3d grid because add_floor==0")
            return
        k = 0
        self.grid[buffer, k] = self.codes["buffer"]
        self.active_codes["buffer"] = True

    def add_paths_to_grid(self, paths):
        k = 0 if not self.add_floor else 1
        for path in paths.values():
            for i, j in (path[0], path[-1]):
                self.grid[i, j, k] = self.codes["node"]
            for i, j in path[1:-1]:
                self.grid[i, j, k] = self.codes["path"]
        self.active_codes["path"] = True
        self.active_codes["node"] = True

    def delete_from_grid(self, code):
        self.grid[self.grid == code] = self.codes["empty"]

    def delete_goals_by_status(self, status):
        for goal_id, goal in self.goals.items():
            if goal["status"] == status:
                self.delete_from_grid(self.goals_codes[goal_id])

    def glyphs_by_code(self, code):
        # Get the indices of occupied cells
        occupied_indices = np.argwhere(self.grid == code)

        if occupied_indices.size == 0:
            print("No occupied cells to display.")
            return

        # Compute the coordinates of the occupied cells (0.5 is the offset to center each cube at the center of each occupied cell)
        x_coords = self.x_min + (occupied_indices[:, 0] + 0.5) * self.resolution
        y_coords = self.y_min + (occupied_indices[:, 1] + 0.5) * self.resolution
        z_coords = self.z_min + (occupied_indices[:, 2] + 0.5) * self.resolution

        # Create a point cloud of the occupied cell centers
        points = np.column_stack((x_coords, y_coords, z_coords))

        # Create a PyVista PolyData object from the points
        point_cloud = pv.PolyData(points)

        # Create a cube to use as the glyph geometry
        cube = pv.Cube(x_length=self.resolution, y_length=self.resolution, z_length=self.resolution)

        # Use glyphs to place a cube at each point
        glyphs = point_cloud.glyph(geom=cube, scale=False, orient=False)

        return glyphs

    def visualize_grid_with_pyvista(self):
        """
        Visualize a 3D occupancy grid using PyVista by placing cubes at occupied cells.

        Returns:
            None: Launches a PyVista plotting window to visualize the grid. Occupied cells are shown as cubes.
        """
        if not is_running_in_jupyter():
            print("pyvista works only in jupyter")
            return

        glyphs_list = []
        for key, code in self.codes.items():
            if self.active_codes[key] is True:
                print(f"Adding {key, code}...")
                glyphs = self.glyphs_by_code(code)
                glyphs_list.append({"code": code, "glyphs": glyphs})

        # Plotting
        print("Plotting...")
        plotter = pv.Plotter()
        for glyphs in glyphs_list:
            plotter.add_mesh(glyphs["glyphs"], show_edges=False, color=self.colors[glyphs["code"]])
        plotter.show()

    def create_2d_map_from_layers(self, obs_codes=None):
        """
        Create a 2D occupancy map from the first nlayers_2d layers above the floor (of add_floor is False, then k starts from 0, else - from 1).

        Returns:
            numpy.ndarray: A 2D occupancy map (I x J). Each cell is 1 if any of
                           the considered layers are occupied at that cell, and 0 otherwise.
        """
        print("Creating 2D occupancy map...", end="")
        # Ensure nlayers_2d is within the bounds
        start = 1 if self.add_floor else 0
        if self.nlayers_2d < start:
            raise ValueError(
                f"Number of layers to merge (nlayers_2d) must be at least {start} to consider layers above the floor.")
        if self.nlayers_2d >= self.grid.shape[2]:
            raise ValueError(
                f"Number of layers to merge (nlayers_2d) is too large; grid only has {self.grid.shape[2]} layers along z.")
        
        if obs_codes is None:
            obs_codes = [self.codes['box'], self.codes['cylinder']]

        # grid shape: (I, J, K) to (I, J, k)
        sliced = self.grid[:, :, start:self.nlayers_2d + start]

        # Filter to keep only values in obs_codes, or empty otherwise
        filtered = np.where(np.isin(sliced, obs_codes), sliced, self.codes['empty'])

        # merged_2d_map = (sliced > 0).any(axis=2).astype(np.uint8) # logical OR
        merged_2d_map = np.max(filtered, axis=2)  # max (considers objects codes)

        self.map_2d = merged_2d_map
        print("\tV")

    def visualize_2d_map(self, map_2d, obstacle_colors=None, show=True) -> object:
        """
        Visualize a 2D map where each cell contains an integer representing an obstacle type.
        Different integers correspond to different obstacle types, and each type is displayed
        with a unique color. The map can be cropped by "offset" cells from each border.

        Parameters:
            obstacle_colors (dict or None): A dictionary mapping each obstacle type (int) to a color (string or RGB tuple).
                                            If None, a default colormap will be generated.
            map_2d (numpy array): obstacles 2D map generated by create_2d_map_from_layers

        Returns:
            None: Displays the 2D map in a matplotlib window.
        """
        if map_2d is None:
            print("Map 2D is not present, generate it first, using create_2d_map_from_layers or add_buffer_zone")
            return

        map_2d = map_2d.T  # flip x and y axes if needed

        # Apply offset (crop the map)
        if self.floor_ext > 0:
            offset = int(self.floor_ext / self.resolution)
            h, w = map_2d.shape
            if offset * 2 >= h or offset * 2 >= w:
                raise ValueError("Invalid offset: too large for the given map dimensions.")
            map_2d = map_2d[offset:-offset, offset:-offset]

        # Identify unique values in the map
        unique_values = sorted(np.unique(map_2d))

        # If no custom color mapping is provided, we generate a colormap automatically
        if obstacle_colors is None:
            value_to_color = {key: color for key, color in self.colors.items() if key in unique_values}
        else:
            # Check all unique_values are in obstacle_colors
            missing = [val for val in unique_values if val not in obstacle_colors]
            if missing:
                raise ValueError(f"Missing color mappings for values: {missing}")
            value_to_color = obstacle_colors
        value_to_color = {key: to_rgba(color) for key, color in value_to_color.items()}

        # Create a list of colors in the order of sorted unique_values
        sorted_values = sorted(unique_values)
        cmap_colors = np.array([value_to_color[val] for val in sorted_values])
        cmap = ListedColormap(cmap_colors)

        # Normalize the data so the smallest unique value maps to index 0 in the colormap
        val_to_idx = {val: i for i, val in enumerate(sorted_values)}
        map_indexed = np.vectorize(val_to_idx.get)(map_2d)

        fig, ax = plt.subplots(figsize=(15, 15))
        ax.imshow(map_indexed, cmap=cmap, interpolation="nearest", origin="lower")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        if show:
            plt.show()
        else:
            return fig, ax

    def add_buffer_zone(self, buffer_length):
        """
        Add a buffer zone around obstacles in a 2D map.

        Parameters:
            buffer_length (int): The radius (in cells) of the buffering zone around obstacles.

        Returns:
            numpy.ndarray: A copy of map_2d with the buffer zone added.
                           Cells that were 0 and lie within "buffer_length" cells of an obstacle
                           are set to "buffer_value".
        """
        print("Adding buffer zone around obstacles...", end="")
        if self.map_2d is None:
            print("Map 2D is not present, generate it first, using create_2d_map_from_layers")
            return
        # Create a binary mask where True = obstacle, False = free space
        obstacle_mask = (self.map_2d > 0)

        # Define the structuring element for dilation
        size = 2 * buffer_length + 1
        struct = np.ones((size, size), dtype=bool)

        # Dilate the obstacle mask
        dilated_mask = binary_dilation(obstacle_mask, structure=struct)

        # Identify which cells were free (0) and now are covered by the dilated region
        buffer_cells = (self.map_2d == 0) & (dilated_mask == True)

        # Create a copy of map_2d to avoid modifying original data
        self.map_2d_buffered = self.map_2d.copy()
        self.map_2d_buffered[buffer_cells] = self.codes["buffer"]
        self.add_buffer_to_grid(buffer_cells)
        print("\tV")

    def map_2d_to_pgm(self, map_name, rotate=0):
        offset = int(self.floor_ext / self.resolution) - 1
        img = Image.fromarray(255 - self.map_2d[offset:-offset, offset:-offset]).rotate(rotate)
        img.save(f"maps/{map_name}.pgm")
        img.show()

    @staticmethod
    def transform_to_local_frame(point, position, rotation):
        """
        Transform a point from the global frame to the local frame of an obstacle.

        Parameters:
            point (tuple): The global (x, y, z) coordinates of the point.
            position (tuple): The global (x, y, z) position of the obstacle"s center.
            rotation (tuple): The (roll, pitch, yaw) rotation of the obstacle.

        Returns:
            np.ndarray: The (x, y, z) coordinates of the point in the obstacle"s local frame.
        """
        translation = np.array(point) - np.array(position)
        rotation_matrix = R.from_euler("xyz", rotation).as_matrix()
        return np.dot(rotation_matrix.T, translation)

    def is_inside_cylinder(self, point, position, rotation, radius, length):
        """
        Check if a point is inside a rotated cylinder.

        Parameters:
            point (tuple): Global (x, y, z) coordinates of the bullet.
            position (tuple): Global (x, y, z) position of the cylinder"s center.
            rotation (tuple): (roll, pitch, yaw) rotation of the cylinder.
            radius (float): Radius of the cylinder.
            length (float): Length of the cylinder along its axis.

        Returns:
            bool: True if the point is inside the rotated cylinder, False otherwise.
        """
        local_point = self.transform_to_local_frame(point, position, rotation)
        x, y, z = local_point
        z_min = -length / 2
        z_max = length / 2
        # print(f"Checking cylinder: Local point: {local_point}, Radius check: {x**2 + y**2 <= radius**2}, Length check: {z_min <= z <= z_max}")
        return (x ** 2 + y ** 2 <= radius ** 2) and (z_min <= z <= z_max)

    def is_inside_rectangle(self, point, position, rotation, dimensions):
        """
        Check if a point is inside a rotated rectangle.

        Parameters:
            point (tuple): Global (x, y, z) coordinates of the bullet.
            position (tuple): Global (x, y, z) position of the rectangle center.
            rotation (tuple): (roll, pitch, yaw) rotation of the rectangle.
            dimensions (tuple): (length, width, height) dimensions of the rectangle.

        Returns:
            bool: True if the point is inside the rotated rectangle, False otherwise.
        """
        local_point = self.transform_to_local_frame(point, position, rotation)
        l, w, h = dimensions
        x, y, z = local_point
        # print(f"Checking rectangle: Local point: {local_point}, Dimensions: {l, w, h}, Check: {(-l/2 <= x <= l/2) and (-w/2 <= y <= w/2) and (-h/2 <= z <= h/2)}")
        return (
                -l / 2 <= x <= l / 2 and
                -w / 2 <= y <= w / 2 and
                -h / 2 <= z <= h / 2
        )

    def calculate_particle_trajectory(self, gun_base, target, pitch_steps=200, gravity=9.81, time_step=0.01, eps=None, exact=False, border=0):
        """
        Calculate a collision-free particle trajectory from start to target, considering obstacles.

        Parameters:
            gun_base (tuple): Starting position of the gun (x, y, z).
            target (tuple): Target position (x, y, z).
            steps (int): Number of pitch steps to try (default is 100).
            gravity (float): Acceleration due to gravity (default is 9.81 m/s²).
            time_step (float): Simulation time step in seconds.
            eps (float): distance from the goal error (default is self.resolution)
            exact: if True - check exact collision, iterating over all obstacles (may be slow). Else - checks collision using the 3d grid

        Returns:
            tuple: A tuple containing:
                - List of trajectory points [(x, y, z), ...] in world coordinates.
                - Rotation angles (pitch, yaw) in radians.
                - Scalar velocity magnitude.
            None: If no collision-free trajectory exists.
        """
        if eps is None:
            eps = self.resolution

        check_collision = self.check_collision_exact if exact else self.check_collision_grid

        dx, dy, dz = np.array(target) - np.array(gun_base)
        yaw = np.arctan2(dy, dx)  # Fixed yaw toward the target

        for pitch in np.linspace(self.max_gun_angle, self.min_gun_angle, pitch_steps):
            # Adjust the start position to the gun tip position
            gun_offset = np.array(self.gun_offset)
            x_tip = gun_offset[0] * np.cos(pitch) * np.cos(yaw) + gun_base[0]
            y_tip = gun_offset[1] * np.cos(pitch) * np.sin(yaw) + gun_base[1]
            z_tip = gun_offset[2] * np.sin(pitch) + gun_base[2]

            start = (x_tip, y_tip, z_tip)
            dx, dy, dz = np.array(target) - np.array(start)

            # equation of initial velocity given pitch and distance data for ballistic motion
            dxy = np.sqrt(dx ** 2 + dy ** 2)
            denominator = (2 * np.cos(pitch) ** 2 * (dxy * np.tan(pitch) - dz))
            if denominator <= 0 or np.isnan(denominator):
                continue
            v0 = np.sqrt((gravity * dxy ** 2) / denominator)
            # print(f"{v0=}")

            if v0 > self.max_particle_speed or v0 <= 0:
                continue  # v0 out of bounds

            # velocity coefficients of axes
            vx = np.cos(pitch) * np.cos(yaw)
            vy = np.cos(pitch) * np.sin(yaw)
            vz = np.sin(pitch)
            vx_actual, vy_actual, vz_actual = v0 * vx, v0 * vy, v0 * vz

            trajectory = []
            x, y, z = start

            while z >= self.z_min:  # Stop if the particle hits the ground
                point = (x, y, z)
                # Check for collisions with obstacles
                collision = check_collision(point, border=border)
                if collision:
                    # print(f"Collision detected on {point=}")
                    break  # Stop trajectory simulation on collision

                trajectory.append(point)

                x += vx_actual * time_step
                y += vy_actual * time_step
                z += vz_actual * time_step - 0.5 * gravity * time_step ** 2
                vz_actual -= gravity * time_step

                if np.linalg.norm([x - target[0], y - target[1], z - target[2]]) < eps:
                    t = dxy / v0
                    return trajectory, round(pitch, self.round_to), round(yaw, self.round_to), round(v0, self.round_to), round(t, self.round_to)  # Valid trajectory found

        return [None] * 5  # No collision-free trajectory found

    def add_trajectory_to_grid(self, trajectory):
        if not trajectory:
            return
        for x, y, z in trajectory:
            i, j, k = self.world_to_grid(x, y, z)
            self.grid[i, j, k] = self.codes["point"]
        self.active_codes["point"] = True

    def add_circle_to_grid(self, points, convert=True):
        if not points:
            return
        k = 1 if self.add_floor is True else 0
        for i, j, _ in points:
            if convert:
                i, j, k = self.world_to_grid(i, j)
            self.grid[i, j, k] = self.codes["circle"]
        self.active_codes["circle"] = True

    def check_collision_exact(self, point, border=None):
        # Check for collisions with obstacles
        for component in self.obstacles:
            for obs in component.values():
                if obs["type"] == "cylinder":
                    if self.is_inside_cylinder(point, obs["position"], obs["orientation"], obs["radius"],
                                               obs["length"]):
                        return True
                elif obs["type"] == "box":
                    if self.is_inside_rectangle(point, obs["position"], obs["orientation"], obs["size"]):
                        return True
        return False

    def check_collision_grid(self, point, border=0):
        indexes_combinations = [self.world_to_grid(*point)]
        if border > 0:
            indexes_combinations = self.generate_combinations(*indexes_combinations[0], border)
        if any(((i < 0 or i >= self.grid.shape[0] or
                    j < 0 or j >= self.grid.shape[1] or
                    k < 0 or k >= self.grid.shape[2] or
                    self.grid[i, j, k] != 0) for i, j, k in indexes_combinations)):
                return True
        return False
    
    @staticmethod
    def generate_combinations(i, j, k, border):
        """
        Generates all combinations of (i, j, k) with variations for each index based on border.

        Args:
            i, j, k (int): Original indices.
            border (int): Positive integer for maximum variation.

        Returns:
            set of tuple: Set of all index combinations.
        """
        combinations = []
        for m in range(1, border + 1):
            variations = [-m, 0, m]
            for x, y, z in product(variations, repeat=3):
                combinations.append((i + x, j + y, k + z))
        
        return set(combinations)

    def calculate_max_height_and_horizontal_distance(self, z_start=0.0, z_finish=0.0, g=9.81):
        """
        Calculate the maximum height and the distance over xy axes for a given velocity and angle.

        Parameters:
        - z_start: Starting height (m).
        - g: Acceleration due to gravity (default: 9.81 m/s^2).

        Returns:
        - max_height: The maximum height reached during the flight (m).
        - xy_distance: The horizontal distance traveled (m).
        """
        v0 = self.max_particle_speed
        angle = self.max_gun_angle
        # Initial velocity components
        vx = v0 * np.cos(angle)
        vy = v0 * np.sin(angle)

        # Maximum height calculation
        max_height = z_start + (vy ** 2) / (2 * g)

        # Time of flight calculation
        t_up = vy / g  # Time to reach max height
        t_down = np.sqrt(2 * (max_height - z_finish) / g)  # Time to fall back down
        total_time = t_up + t_down

        # Horizontal distance (xy distance)
        xy_distance = vx * total_time

        return max_height, xy_distance

    def generate_circle_samples(self, center, radius, num_samples):
        """
        Generate grid indices for points on a circle on obstacle-free area.

        Parameters:
            center (tuple): The (row, col) indices of the circle's center in the grid.
            radius (float): The radius of the circle in meters.
            num_samples (int): Number of samples (cells) to return on the circle.

        Returns:
            list: A list of tuples representing the (row, col) indices of sampled points on the circle.
        """
        center_row, center_col, _ = self.world_to_grid(center[0], center[1])
        grid_rows, grid_cols = self.map_2d_buffered.shape
        radius_cell = radius / self.resolution

        samples = []
        # Generate samples evenly spaced on the circle
        for i in range(num_samples):
            angle_rad = 2 * np.pi * i / num_samples

            # Calculate the relative position on the circle in cell coordinates
            rel_row = radius_cell * np.cos(angle_rad)
            rel_col = radius_cell * np.sin(angle_rad)
            # Map the relative position to absolute grid indices
            circle_row = round(center_row + rel_row)
            circle_col = round(center_col + rel_col)

            # Check if the calculated indices are within grid bounds and not on obstacles
            if 0 <= circle_row < grid_rows and 0 <= circle_col < grid_cols and self.map_2d_buffered[circle_row, circle_col] == 0:
                samples.append((*self.grid_to_world(circle_row, circle_col)[:2], self.gun_joint_offset[2]))
        return samples
