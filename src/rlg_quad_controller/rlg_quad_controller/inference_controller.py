import rclpy
from rclpy.node import Node

import yaml
import torch
import numpy as np
import transforms3d as t3d

from sensor_msgs.msg import Imu, JointState
from geometry_msgs.msg import Twist
from pi3hat_moteus_int_msgs.msg import JointsCommand, JointsStates

from .utils.torch_utils import quat_rotate_inverse_numpy
from .utils.rlg_utils import build_rlg_model, run_inference


class InferenceController(Node):
    """
    ROS2 node per eseguire inference con una policy rl‑games
    e pubblicare comandi articolari a rate fissa.
    """

    def __init__(self):
        super().__init__('inference_controller')

        # --- Dichiarazione parametri ROS ---
        self.declare_parameter('model_path', '')
        self.declare_parameter('env_cfg_path', '')
        self.declare_parameter('agent_cfg_path', '')
        self.declare_parameter('simulation', False)
        self.declare_parameter('joint_state_topic', '/joint_states')
        self.declare_parameter('joint_target_topic', '/target_joint_states')
        self.declare_parameter('cmd_vel_topic', '/teleop_twist_keyboard')
        # nuovi parametri per scaling
        self.declare_parameter('angular_velocity_scale', 1.0)
        self.declare_parameter('cmd_vel_scale', 1.0)

        # --- Lettura parametri ---
        self.model_path       = self.get_parameter('model_path').value
        self.env_cfg_path     = self.get_parameter('env_cfg_path').value
        self.agent_cfg_path   = self.get_parameter('agent_cfg_path').value
        self.simulation       = self.get_parameter('simulation').value
        self.joint_state_topic= self.get_parameter('joint_state_topic').value
        self.joint_target_topic = self.get_parameter('joint_target_topic').value
        self.cmd_vel_topic    = self.get_parameter('cmd_vel_topic').value
        self.angular_vel_scale= self.get_parameter('angular_velocity_scale').value
        self.cmd_vel_scale    = self.get_parameter('cmd_vel_scale').value

        # --- Caricamento YAML di configurazione ---
        with open(self.env_cfg_path, 'r') as f:
            self.env_cfg = yaml.load(f, Loader=yaml.Loader)
        with open(self.agent_cfg_path, 'r') as f:
            self.agent_cfg = yaml.load(f, Loader=yaml.Loader)

        # --- Calcolo rate di inference ---
        dt = self.env_cfg['sim']['dt']
        decimation = self.env_cfg['decimation']
        self.rate_hz = 1.0 / (dt * decimation)
        self.get_logger().info(f'Inference rate: {self.rate_hz:.2f} Hz')

        # --- Scaling azioni ---
        leg_scale   = self.env_cfg['actions']['joint_pos']['scale']
        wheel_scale = self.env_cfg['actions']['joint_vel']['scale']
        self.action_scale = np.array([leg_scale]*8 + [wheel_scale]*4).reshape((12,1))

        # --- Caricamento modello RL‑Games ---
        self.get_logger().info(f"Loading rl‑games checkpoint: {self.model_path}")
        self.model = build_rlg_model(
            weights_path   = self.model_path,
            env_cfg_path   = self.env_cfg_path,
            agent_cfg_path = self.agent_cfg_path,
            device         = 'cuda:0'
        )
        self.get_logger().info('Model successfully loaded.')

        # --- Pubblicatori/Sottoscrittori ---
        if self.simulation:
            self.joint_pub = self.create_publisher(JointState, self.joint_target_topic, 10)
            self.joint_sub = self.create_subscription(
                JointState, self.joint_state_topic, self.joint_state_callback, 10
            )
        else:
            self.joint_pub = self.create_publisher(JointsCommand, self.joint_target_topic, 10)
            self.joint_sub = self.create_subscription(
                JointsStates, self.joint_state_topic, self.joint_state_callback, 10
            )
        self.cmd_sub = self.create_subscription(
            Twist, self.cmd_vel_topic, self.cmd_vel_callback, 10
        )

        self.declare_parameter('use_imu', False)
        self.use_imu = self.get_parameter('use_imu').value
        if self.use_imu:
            self.declare_parameter('imu_topic', '/IMU_Broadcaster/imu')
            imu_topic = self.get_parameter('imu_topic').value
            self.imu_sub = self.create_subscription(Imu, imu_topic, self.imu_callback, 10)
        else:
            # inizializzo a zero se non uso IMU
            self.base_ang_vel = np.zeros((3,1))
            self.projected_gravity = np.zeros((3,1))

        # --- Buffer di stato ---
        self.n_joints = 12
        self.joint_names = [
            'LF_HFE','LH_HFE','RF_HFE','RH_HFE',
            'LF_KFE','LH_KFE','RF_KFE','RH_KFE',
            'LF_WHEEL_JNT','LH_WHEEL_JNT','RF_WHEEL_JNT','RH_WHEEL_JNT'
        ]
        self.joint_pos = {n:0.0 for n in self.joint_names}
        self.joint_vel = {n:0.0 for n in self.joint_names}
        self.prev_action = np.zeros((self.n_joints,1))

        # --- Posa di default e warmup ---
        hip  = np.deg2rad(120.0)
        knee = np.deg2rad(60.0)
        self.default_pose = np.array([
            hip, -hip, -hip, hip,
            -knee, knee, knee, -knee,
            0,0,0,0
        ])
        self._warmup_duration = 3.0
        self.start_time = self.get_clock().now()

        # --- Timer di inference ---
        self.timer = self.create_timer(1.0/self.rate_hz, self.inference_callback)
        self.get_logger().info('Node initialized and ready.')

    def cmd_vel_callback(self, msg: Twist):
        self.cmd_vel = np.array([
            msg.linear.x, msg.linear.y, msg.angular.z
        ]).reshape((3,1))

    def imu_callback(self, msg: Imu):
        """
        Update base angular velocity and orientation from IMU.
        Project gravity vector into robot frame.
        """
        self.base_ang_vel = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ]).reshape((3, 1))

        quat = [
            msg.orientation.w,
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z
        ]
        yaw, pitch, roll = t3d.euler.quat2euler(quat, axes='szyx')
        corrected = t3d.euler.euler2quat(
            self.yaw_bias, pitch, roll, axes='szyx'
        )
        self.base_quat = np.array(corrected).reshape((4, 1))
        self.projected_gravity = quat_rotate_inverse_numpy(
            self.base_quat, np.array([0, 0, -1]).reshape((3, 1))
        )

    def joint_state_callback(self, msg: JointState):
        """
        Update current joint positions and velocities.
        """
        for name, pos, vel in zip(msg.name, msg.position, msg.velocity):
            if name in self.joint_pos:
                self.joint_pos[name] = pos
                self.joint_vel[name] = vel

    def inference_callback(self):
        now = self.get_clock().now()
        # costruisco l’osservazione
        obs = np.vstack([
            self.base_ang_vel * self.angular_vel_scale,
            self.projected_gravity,
            self.cmd_vel * self.cmd_vel_scale,
            np.fromiter(self.joint_pos.values(), dtype=float).reshape((self.n_joints,1)), 
            np.fromiter(self.joint_vel.values(), dtype=float).reshape((self.n_joints,1)), 
            self.prev_action
        ]).reshape((1,-1))

        action = run_inference(self.model, obs, det=True).flatten()
        self.prev_action = action.reshape((self.n_joints,1))

        # warmup default pose
        delta = now - self.start_time
        elapsed = delta.seconds + delta.nanoseconds * 1e-9
        if elapsed < self._warmup_duration:
            target = self.default_pose
        else:
            target = action * self.action_scale.flatten() + self.default_pose

        # pubblicazione
        if self.simulation:
            msg = JointState()
        else:
            msg = JointsCommand()
            msg.kp_scale = [1.0]*self.n_joints
            msg.kd_scale = [1.0]*self.n_joints

        msg.header.stamp = now.to_msg()
        msg.name = self.joint_names
        msg.position = target.tolist()
        self.joint_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = InferenceController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()