import os
import yaml

from launch import LaunchDescription,LaunchContext
from launch.events.process import ProcessStarted
from launch.actions import DeclareLaunchArgument,RegisterEventHandler,ExecuteProcess,LogInfo,EmitEvent
from launch.substitutions import Command,LaunchConfiguration
from launch.event_handlers import OnProcessStart,OnProcessIO,OnProcessExit,OnShutdown,OnExecutionComplete

from ament_index_python.packages import get_package_share_path
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
import xml.etree.ElementTree as ET
import subprocess
import time


def generate_launch_description():
    joy_node_path = get_package_share_path("omni_mulinex_joystic")

    joy_cfg_file = os.path.join(joy_node_path,"config","joy_node.yaml")

    # subprocess.check_output(
    #     ["ros2 control load_controller omni_control --set-state active "]
    #     ,shell=True)
    
    # subprocess.check_output(
    #     ["ros2 control load_controller joint_state_broadcaster --set-state active"]
    #     ,shell=True)

    joy_event_node = Node(
        package="joy",
        executable="joy_node",
        output="screen"
    ) 

    joy_node = Node(
        package="omni_mulinex_joystic",
        executable="omni_mul_joystic_node",
        output="screen",
        parameters=[joy_cfg_file]
    )
    return LaunchDescription([
        joy_event_node,
        joy_node
    ])