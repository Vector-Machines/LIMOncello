cmake_minimum_required(VERSION 3.16)
project(limoncello LANGUAGES CXX)

# Enable ccache if available
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
endif()

# Default build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Require C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -pthread -fexceptions -O3 -lm -lblas -llapack")
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# add_compile_options(-g)

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
find_package(PCL REQUIRED)
find_package(manif REQUIRED)

# Define executable
add_executable(${PROJECT_NAME} src/main.cpp)

# Include directories
target_include_directories(${PROJECT_NAME}
  PRIVATE
    include
    ${PCL_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${manif_INCLUDE_DIRS}
)

# Compiler options
target_compile_options(${PROJECT_NAME}
  PRIVATE
    $<$<CONFIG:Release>:-O3;-march=native>
    -pthread
    -fexceptions
)

# Link libraries with explicit keyword (PRIVATE)
target_link_libraries(${PROJECT_NAME}
  PRIVATE
    ${PCL_LIBRARIES}
    ${Boost_LIBRARIES}
    ${TBB_IMPORTED_TARGETS}
)

# ROS2 dependencies (ament_target_dependencies)
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

# Install rules
install(TARGETS ${PROJECT_NAME} DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})

ament_package()
