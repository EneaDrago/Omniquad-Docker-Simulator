cmake_minimum_required(VERSION 3.8)
project(mulinex_ignition)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(mulinex_description REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(pi3hat_moteus_int_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(control_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(ament_index_cpp REQUIRED)

add_executable(gt_odom_pub src/gt_odom_pub.cpp)
ament_target_dependencies(
    gt_odom_pub
    geometry_msgs
    nav_msgs
    rclcpp
    tf2
    tf2_ros
)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
install(TARGETS
    gt_odom_pub
    DESTINATION lib/${PROJECT_NAME})

# move_wheels node
add_executable(move_wheels src/move_wheels.cpp)
ament_target_dependencies(
    move_wheels
    geometry_msgs
    nav_msgs
    rclcpp
    tf2
    tf2_ros
    pi3hat_moteus_int_msgs
    std_srvs
)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
install(TARGETS
    move_wheels
    DESTINATION lib/${PROJECT_NAME})

# getup_node
add_executable(getup_node
  src/getup.cpp
)
ament_target_dependencies(getup_node
  rclcpp
  control_msgs
)
install(TARGETS getup_node
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch config  world  models
      DESTINATION share/${PROJECT_NAME}
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

ament_package()
