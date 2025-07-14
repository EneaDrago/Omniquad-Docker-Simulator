import os
from ament_index_python.packages import get_package_share_path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import Command,LaunchConfiguration,PathJoinSubstitution
from launch.conditions import IfCondition,UnlessCondition

from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():

    #path for build mulinex12 urdf
    mulinex_robot_path = get_package_share_path("mulinex_description")
    mulinex_robot_path = os.path.join(mulinex_robot_path,"urdf", "mulinex.xacro") 
    # #path for rviz settings
    rviz_config_path = get_package_share_path("mulinex_description")
    rviz_config_path = os.path.join(rviz_config_path, "rviz", "config.rviz")
    # world file
    world_file = get_package_share_path("mulinex_description")
    world_file = os.path.join(
        # world_file, "world", "empty.world"
        # world_file,"world","obs_up.world"
        # world_file,"world","obs_down.world"
        world_file,"world","gap.world"
    )

    #declaration argument of launch
    use_gui = DeclareLaunchArgument(
        name="use_gui",
        default_value="true",
        description="Value use to enable joint publisher with GUI")

    mulinex_model = DeclareLaunchArgument(
        name="mulinex_urdf",
        default_value=str(mulinex_robot_path)
    )

    # rviz_arg = DeclareLaunchArgument(
    #     name="rviz_config",
    #     default_value= str(rviz_config_path),
    #     description="configuration of Rviz for plot"
    # )

    #use command to create a parameter with urdf of mulinex by xacro file
    robot_description = ParameterValue(
        Command(["xacro ",LaunchConfiguration("mulinex_urdf")]),
        value_type=str
    )

    #node declaration
    robot_state_pub = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description}]
    )
    
    #launch gazebo 
    gazebo = ExecuteProcess(
        cmd=['gazebo', '--verbose','--pause' ,'-s', 'libgazebo_ros_factory.so', world_file], 
        output='screen'
    )

    spawn_entity = Node(
        package='gazebo_ros', 
        executable='spawn_entity.py',
        arguments=['-topic', 'robot_description', '-entity', 'mulinex', '-x', '0', '-y', '0', '-z', '0.12'],
        output='screen'
    )

    return LaunchDescription(
        [
            mulinex_model,
            robot_state_pub,
            gazebo,
            spawn_entity
        ]
    )
