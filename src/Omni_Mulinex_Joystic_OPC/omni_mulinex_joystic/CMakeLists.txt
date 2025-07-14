cmake_minimum_required(VERSION 3.8)
project(omni_mulinex_joystic)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(pi3hat_moteus_int_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(rosbag2_cpp REQUIRED)

set(exec_name ${PROJECT_NAME}_exec)

#add node to library 
add_library(
  ${PROJECT_NAME} SHARED
  src/omnimul_joy.cpp
)

# add include 
target_include_directories(
  ${PROJECT_NAME}
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

#add dependent package
ament_target_dependencies(
  ${PROJECT_NAME}
  rclcpp
  sensor_msgs
  pi3hat_moteus_int_msgs
  std_msgs
  std_srvs
  rosbag2_cpp
)

add_executable(
  omni_mul_joystic_node
  src/om_joy_main.cpp
)

target_link_libraries(omni_mul_joystic_node 
PUBLIC
${PROJECT_NAME})

install(TARGETS
omni_mul_joystic_node
  DESTINATION lib/${PROJECT_NAME})

#install include
install(
  DIRECTORY include
  DESTINATION include/${PROJECT_NAME}
)

install(
  TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)


# # link library 
# target_link_libraries(
#   rclcpp:Node
# )



install(
  DIRECTORY launch config
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

ament_export_dependencies(
  rclcpp
  sensor_msgs
  pi3hat_moteus_int_msgs
  std_msgs
  std_srvs
  rosbag2_cpp
)
ament_export_include_directories(
  include
)
ament_export_libraries(
  ${PROJECT_NAME}
)



ament_package()
