# Tank robot project

A robo-tank with water gun that patrols your backyard and annoys street cats, firing (a warm) water on them (summer only).
------------------

## Installation
### Installing rabot_tank package
(Assuming you have your catking workspace named catkin_ws)

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

OR
Or using PyPi to install all the requirements:
```sh
pip install -r ~/catkin_ws/src/tank_robot/requirements.txt
```
