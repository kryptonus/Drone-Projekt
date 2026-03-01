cmake_minimum_required(VERSION 3.8)
project(quadcopter_control)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(gazebo_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(include)

add_executable(face_tracker_node src/face_tracker_node.cpp)

ament_target_dependencies(face_tracker_node
  rclcpp
  gazebo_msgs
  geometry_msgs
  sensor_msgs
  cv_bridge
)

target_link_libraries(face_tracker_node
  ${OpenCV_LIBS}
)

install(TARGETS face_tracker_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include/
)

ament_package()