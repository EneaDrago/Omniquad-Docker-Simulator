import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    ld = LaunchDescription()

    params = os.path.join(
        get_package_share_directory('rlg_quad_controller'),
        'config',
        'mulinex_config.yaml'
        )
    
    config_path = os.path.join(
        get_package_share_directory('rlg_quad_controller'),
        'models',
        'omni_flat_v0',
        'config.yaml'
        )
    
    weights_path = os.path.join(
        get_package_share_directory('rlg_quad_controller'),
        'models',
        'omni_flat_v0',
        'omniquad_flat.pth'
        )
    
    node=Node(
        package = 'rlg_quad_controller',
        name = 'inference_controller',
        executable = 'inference_controller',
        parameters = [params,
                      {'config_path': config_path},
                      {'model_path': weights_path}]
    )

    ld.add_action(node)
    return ld