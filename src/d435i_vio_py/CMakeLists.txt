cmake_minimum_required(VERSION 3.0.2)
project(d435i_vio_py)

## Find catkin and system dependencies
find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  cv_bridge           # <-- important
)

# Find OpenCV
find_package(OpenCV REQUIRED)  # <-- important

catkin_package(
  CATKIN_DEPENDS roscpp sensor_msgs cv_bridge
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}       # <-- important
)

add_executable(d435i_preprocessed_viewer_node
  src/d435i_preprocessed_viewer.cpp
)

target_link_libraries(d435i_preprocessed_viewer_node
  ${catkin_LIBRARIES}
  # ${OpenCV_LIBRARIES}          # <-- critical line
  ${OpenCV_LIBS}
)
