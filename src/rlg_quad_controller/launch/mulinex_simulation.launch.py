import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
import launch_ros.actions

def generate_launch_description():

    params = os.path.join(
        get_package_share_directory('rlg_quad_controller'),
        'config',
        'mulinex_simulation_config.yaml'
        )
    # print("************** USING FEEDFORWARD CONFOIG************** ")
    # FIXME: pass model folder as command line argument

    config_path = os.path.join(
        get_package_share_directory('rlg_quad_controller'),
        'models',
        'RealGaitRew2NoHeading',
        'config.yaml'
        )
    
    weights_path = os.path.join(
        get_package_share_directory('rlg_quad_controller'),
        'models',
        'RealGaitRew2NoHeading',
        'RealGaitRew2NoHeading.pth'
        )
    
    node=Node(
        package = 'rlg_quad_controller',
        name = 'inference_controller',
        executable = 'inference_controller',
        parameters = [params,
                      {'config_path': config_path},
                      {'model_path': weights_path}]
    )

    return LaunchDescription([
        launch_ros.actions.SetParameter(name='use_sim_time', value=False),
        node])

