# Tank robot project

A robo-tank with water gun that patrols your backyard and annoys street cats, firing (a warm) water on them (summer only).
------------------

## Installation
### Installing rabot_tank package
(Assuming you have your catkin workspace named catkin_ws)

```sh
cd ~/catkin_ws/src
git clone https://github.com/VladKlimen/tank_robot.git
cd ~/catkin_ws
rosdep install --from-paths src --ignore-src -r -y
catkin_make
source ~/catkin_ws/devel/setup.bash
```

### Installing python requirements
Conda (new environment):
```sh
conda env create -f ~/<your_catkin_ws>/src/tank_robot/tank_robot_environment.yml
```

Or using PyPi to install all the requirements:
```sh
pip install -r ~/catkin_ws/src/tank_robot/requirements.txt
```

## Running simulations
Aviable worlds names: 
by_small (default world) - no unreachable obstacle-free driving destinations.

by_small2 - there is an obstacle that divides the back yard in sections that are unreachable (by driving) on from another.

Run in first ros terminal:
```sh
roslaunch tank_robot gazebo.launch
```
To change the world to by_small2:
```sh
roslaunch tank_robot gazebo.launch world_name:=by_small2
```

Run in second ros terminal:
```sh
roscd tank_robot/scripts
./main_tank_routine.py
```

The simulation should start to run.

## Manual control
You can control the tank and the gun manually, by running (along with the gazebo.launch in a different terminal):
```sh
roscd tank_robot/scripts
./teleop_tank_keyboard.py
```

## Visualizing plan
See the plan_visualization.ipyb and the plan_visualization_examples folder
