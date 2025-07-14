import imp
import os
from ament_index_python.packages import get_package_share_path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command,LaunchConfiguration,PathJoinSubstitution
from launch.conditions import IfCondition,UnlessCondition

from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():

    #path for build mulinex12 urdf
    

    base_mot_node = Node(
        package='ik_quad_foot_control',
        executable='base_motion',
        name='base_motion_node',
        output= 'screen',
        arguments=[
                {'amplitude': 0.01},
                {'frequency': 0.1},
                {'motion': 1},
                {'use_sim_time': True},
                {'x_pos_bound': 0.25},
                {'z_pos_bound': 0.35}
              ],
    )



    return LaunchDescription(
        [
           
		base_mot_node
        ]
    )
