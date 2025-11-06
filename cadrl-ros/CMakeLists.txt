cmake_minimum_required(VERSION 3.5)
project(cadrl_ros)

# Use C++14 (recommended for ROS 2 Foxy+)
# Uncomment if you have C++ code
# add_compile_options(-std=c++14)

# Find ament and dependencies

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(ford_msgs REQUIRED)

find_package(std_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
# Include directories (if you have find_package(ford_msgs REQUIRED)headers)
# include_directories(include)

# Example for a C++ node (uncomment if needed)
# add_executable(${PROJECT_NAME}_node src/cadrl_ros_node.cpp)
# ament_target_dependencies(${PROJECT_NAME}_node
#   rclcpp
#   geometry_msgs
#   std_msgs
#   visualization_msgs
#   ford_msgs
# )

# # Install C++ executable (if used)
# install(TARGETS
#   ${PROJECT_NAME}_node
#   DESTINATION lib/${PROJECT_NAME}
# )

# Install Python scripts (ROS 2 uses lib/${PROJECT_NAME})
install(PROGRAMS
  scripts/cadrl_node.py
  DESTINATION lib/${PROJECT_NAME}
 )

# Install launch files or other resources
install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL
)

# Export package
ament_package()
