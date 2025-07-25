cmake_minimum_required(VERSION 3.8)
project(ik_quad_foot_control)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(controller_interface REQUIRED)
find_package(pluginlib REQUIRED)
find_package(pi3hat_moteus_int_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(sensor_msgs REQUIRED)


add_library(
  ${PROJECT_NAME}
  SHARED src/ik_quad_foot_control.cpp
)

add_executable(
	base_motion src/base_twist_cmd.cpp
)
target_include_directories(
  ${PROJECT_NAME} PRIVATE include
)

ament_target_dependencies(${PROJECT_NAME}
  builtin_interfaces
  controller_interface
  hardware_interface
  pluginlib
  rclcpp_lifecycle
  rclcpp
  sensor_msgs
  pi3hat_moteus_int_msgs
  std_srvs
)

pluginlib_export_plugin_description_file(controller_interface ik_quad_foot_control.xml)

target_include_directories(${PROJECT_NAME} PUBLIC

$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>

$<INSTALL_INTERFACE:include>)

install(
  TARGETS
  ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
)


ament_target_dependencies(
  base_motion
  rclcpp
  pi3hat_moteus_int_msgs
  std_srvs
  sensor_msgs
)


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()
ament_export_dependencies(
  builtin_interfaces
  controller_interface
  hardware_interface
  pluginlib
  rclcpp_lifecycle
  rclcpp
  sensor_msgs
  pi3hat_moteus_int_msgs
  std_srvs
  
)
ament_export_include_directories(
  include 
)
ament_export_libraries(
  ${PROJECT_NAME}
)

install(TARGETS
  base_motion
  DESTINATION lib/${PROJECT_NAME})
  
  install(
    DIRECTORY launch 
    DESTINATION share/${PROJECT_NAME}
  )

ament_package()
