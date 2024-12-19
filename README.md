# Tank robot project
## A robo-tank with water gun that patrols your backyard and annoys street cats, firing (a warm) water on them (summer only).

## Installation
(Assuming you have your catking workspace named catkin_ws)
cd ~/catkin_ws/src
git clone https://github.com/VladKlimen/tank_robot.git
cd ~/catkin_ws
rosdep install --from-paths src --ignore-src -r -y
catkin_make
source ~/catkin_ws/devel/setup.bash

### Installing python requirements
Conda (new environment):
conda env create -f ~/<your_catkin_ws>/src/tank_robot/tank_robot_environment.yml

OR

Using pip:
pip install -r ~/<your_catkin_ws>/src/tank_robot/requirements.txt
