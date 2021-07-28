# Potential_function
This is a Potential function code. You can make Potential Map and display Waypoints with Potential values.
# Install dependent packages
```
pip install PyQt5
pip install mayavi
pip install open3d
pip install numpy
pip install pyyaml
```
# Usage
### 1. Prepare map.pcd and waypoint.csv
### 2. Modify ["pcd_data"] and ["csv_waypoint"] in config.yaml to your {map.pcd} & {waypoint.csv}
### 3-1. Make a Potential Map and display potential map and waypoints
```
python potential.py
```
This process takes while and make a potential_value.json
### 3-2. If you only want to display Potential Map and wayoints
```
python display_potential.py
```
# Photo
![first](https://github.com/Chotaro-0322/Potential_function/wiki/image/first.png)
![second](https://github.com/Chotaro-0322/Potential_function/wiki/image/second.png)
![third](https://github.com/Chotaro-0322/Potential_function/wiki/image/third.png)
