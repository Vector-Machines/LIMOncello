cmake_minimum_required(VERSION 3.5)
project(limoncello)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (Release or Debug)" FORCE)
endif()

# Require C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -pthread -fexceptions -O3 -lm -lblas -llapack")

# ROS2 dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)

# Other dependencies
find_package(Boost REQUIRED COMPONENTS system)
find_package(TBB REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL 1.8 REQUIRED)
find_package(manif REQUIRED)

# Find ROS packages
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)

# Include directories
include_directories(
  include
)

# Create executable
add_executable(${PROJECT_NAME} src/main.cpp)

ament_target_dependencies(${PROJECT_NAME} 
  PUBLIC 
    rclcpp 
    pcl_conversions
    sensor_msgs 
    nav_msgs 
    tf2 
    tf2_ros 
    tf2_eigen
)

# Link the executable with libraries
target_link_libraries(${PROJECT_NAME} 
  PUBLIC 
    ${PCL_LIBRARIES}
    ${Boost_LIBRARIES}
    ${TBB_IMPORTED_TARGETS}
    ${manif_INCLUDE_DIRS}
)

install(TARGETS 
  ${PROJECT_NAME}
  DESTINATION lib/${PROJECT_NAME} 
)


install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()