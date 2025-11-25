
### DEPENDENCIES

Follow the Realsense tutorials. This build currently does not build anything from source.

realsense-ros
opencv-bridge

```
sudo apt install librealsense2-utils
sudo apt install librealsense2-dev
sudo apt install librealsense2-dbg
```


For opencv to show up in pylance you may need to install the stubs 
python3 -m pip install --user opencv-stubs
Ctrl+Shift+P → Developer: Reload Window
Ctrl+Shift+P → Pylance: Restart Language Server


### GOTCHAS
DO not source workspaces when launching realsense2_camera -- only source ROS

Check opencv versions on the machine
```apt list --installed | grep opencv```