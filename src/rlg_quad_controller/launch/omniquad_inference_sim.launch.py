import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('rlg_quad_controller')
    params    = os.path.join(pkg_share, 'config', 'mulinex_config.yaml')
    env_path  = os.path.join(pkg_share, 'models', 'omni_flat_v1', 'env.yaml')
    agent_path= os.path.join(pkg_share, 'models', 'omni_flat_v1', 'agent.yaml')
    model_path   = os.path.join(pkg_share, 'models', 'omni_flat_v1', 'omniquad_flat.pth')

    node = Node(
        package    = 'rlg_quad_controller',
        executable = 'inference_controller_sim',
        name       = 'inference_controller_sim',
        parameters = [
            params,
            {'env_cfg_path':    env_path},
            {'agent_cfg_path':  agent_path},
            {'model_path':      model_path},
            {'use_imu':         True},
            {'imu_topic':       '/mulinex/imu'},
            {'joint_state_topic':       '/Joint_Feedback'},
            # se vuoi, puoi aggiungere override per gli scale:
            # {'angular_velocity_scale': 1.0},
            # {'cmd_vel_scale':         1.0},
        ]
    )

    return LaunchDescription([node])