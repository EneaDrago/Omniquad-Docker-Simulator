# bringup_omniquad.launch.py
from launch import LaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
import datetime

'''
File di launch per avviare:
- simulazione Gazebo/IGN
- nodo di inference RL
- teleop da tastiera
- bridge per i topic da quelli nostri ai quelli del robot
- rosbag per registrare tutto
'''

def generate_launch_description():
    # Path al launch di Gazebo/IGN
    ign_pkg = get_package_share_directory('mulinex_ignition')
    ign_launch_path = os.path.join(
        ign_pkg, 'launch', 'gz_harmonic_sim_w_rbt_PD_wheels.launch.py'
    )

    # Include del launch di simulazione
    sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(ign_launch_path)
    )

    inference_pkg = get_package_share_directory('rlg_quad_controller')
    inference_launch_path = os.path.join(
        inference_pkg, 'launch', 'omniquad_inference_sim.launch.py'
    )

    inference_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(inference_launch_path)
    )


    # Teleop da tastiera (richiede input da terminale)
    teleop_node = Node(
        package='teleop_twist_keyboard',
        executable='teleop_twist_keyboard',
        name='teleop_twist_keyboard',
        output='screen',
        prefix=['xterm -e '],
    )

    # Bridge per i topic (da quelli nostri a quelli del robot)
    bridge_node = Node(
        package='mulinex_ignition_py',
        executable='bridge_node',
        name='bridge_node',
        output='screen'
    )


     # --- Directory rosbag con timestamp ---
    rosbag_root = os.path.join(os.getcwd(), "rosbag")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    rosbag_dir = os.path.join(rosbag_root, f"run_{timestamp}")
    os.makedirs(rosbag_root, exist_ok=True)  # assicura la cartella padre

    # Registrazione rosbag in formato MCAP dentro ./rosbag/<timestamp>/
    record_process = ExecuteProcess(
        cmd=["ros2", "bag", "record", "-a", "-s", "mcap", "-o", rosbag_dir],
        output="screen"
    )


    return LaunchDescription([
        sim_launch,
        inference_launch,
        teleop_node,
        bridge_node,
        record_process,

    ])
