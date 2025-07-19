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
    ROS2 node for running inference with an rl-games policy
    and publishing joint commands at a fixed rate.
    """

    def __init__(self):
        super().__init__('inference_controller')

        # --- Declare and read ROS parameters ---
        self.declare_parameter('model_path', '')
        self.declare_parameter('env_cfg_path', '')
        self.declare_parameter('agent_cfg_path', '')
        self.declare_parameter('simulation', False)
        self.declare_parameter('joint_state_topic', '/joint_states')
        self.declare_parameter('joint_target_topic', '/target_joint_states')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')

        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.env_cfg_path = self.get_parameter('env_cfg_path').get_parameter_value().string_value
        self.agent_cfg_path = self.get_parameter('agent_cfg_path').get_parameter_value().string_value
        self.simulation = self.get_parameter('simulation').get_parameter_value().bool_value
        self.joint_state_topic = self.get_parameter('joint_state_topic').get_parameter_value().string_value
        self.joint_target_topic = self.get_parameter('joint_target_topic').get_parameter_value().string_value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').get_parameter_value().string_value

        # --- Load configs ---
        with open(self.env_cfg_path, 'r') as f:
            self.env_cfg = yaml.load(f, Loader=yaml.Loader)
        with open(self.agent_cfg_path, 'r') as f:
            self.agent_cfg = yaml.load(f, Loader=yaml.Loader)

        # --- Compute inference rate ---
        dt = self.env_cfg['sim']['dt']
        decimation = self.env_cfg['decimation']
        self.rate_hz = 1.0 / (dt * decimation)
        self.get_logger().info(f'Inference rate: {self.rate_hz:.2f} Hz')

        # --- Action scaling ---
        leg_scale = self.env_cfg['actions']['joint_pos']['scale']
        wheel_scale = self.env_cfg['actions']['joint_vel']['scale']
        self.action_scale = np.array([leg_scale] * 8 + [wheel_scale] * 4).reshape((12, 1))

        # --- Observation scaling ---
        # clip_actions stored under agent_cfg->params->env->clip_actions
        self.dof_position_scale = float(
            self.agent_cfg['params']['env']['clip_actions']
        )
        # velocity scale not exposed; set to a reasonable default or read from config
        self.dof_velocity_scale = self.env_cfg['learn'].get('dofVelocityScale', 0.05)

        # --- Load trained model ---
        self.get_logger().info(f"Loading rl-games checkpoint: {self.model_path}")
        self.model = build_rlg_model(
            weights_path=self.model_path,
            env_cfg_path=self.env_cfg_path,
            agent_cfg_path=self.agent_cfg_path,
            device='cuda:0'
        )
        self.get_logger().info('Model successfully loaded.')

        # --- Initialize publishers/subscribers ---
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
        self.use_imu = self.get_parameter('use_imu').get_parameter_value().bool_value
        if self.use_imu:
            self.declare_parameter('imu_topic', '/IMU_Broadcaster/imu')
            imu_topic = self.get_parameter('imu_topic').get_parameter_value().string_value
            self.imu_sub = self.create_subscription(Imu, imu_topic, self.imu_callback, 10)

        # --- State buffers ---
        self.n_joints = 12
        self.joint_names = [
            'LF_HFE','LH_HFE','RF_HFE','RH_HFE',
            'LF_KFE','LH_KFE','RF_KFE','RH_KFE',
            'LF_WHEEL_JNT','LH_WHEEL_JNT','RF_WHEEL_JNT','RH_WHEEL_JNT'
        ]
        self.joint_pos = {name: 0.0 for name in self.joint_names}
        self.joint_vel = {name: 0.0 for name in self.joint_names}
        self.prev_action = np.zeros((self.n_joints, 1))

        # --- Default pose ---
        hip = np.deg2rad(120.0)
        knee = np.deg2rad(60.0)
        self.default_pose = np.array([
            hip, -hip, -hip, hip,
            -knee, knee, knee, -knee,
            0, 0, 0, 0
        ])
        self._warmup_duration = 3.0  # seconds to hold default pose at startup
        self.start_time = self.get_clock().now()

        # --- Start inference timer ---
        self.timer = self.create_timer(1.0 / self.rate_hz, self.inference_callback)
        self.get_logger().info('Node initialized and ready.')

    def cmd_vel_callback(self, msg: Twist):
        """
        Store the latest velocity command.
        """
        self.cmd_vel = np.array([
            msg.linear.x, msg.linear.y, msg.angular.z
        ]).reshape((3, 1))
        # clamp if needed

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
        """
        Build observation, run inference, and publish joint targets.
        """
        # concatenate observation vector
        obs = np.vstack([
            self.base_ang_vel * self.angularVelocityScale,
            self.projected_gravity,
            (self.cmd_vel * self.cmd_vel_scale),
            np.fromiter(self.joint_pos.values(), dtype=float).reshape((self.n_joints, 1)) * self.dof_position_scale,
            np.fromiter(self.joint_vel.values(), dtype=float).reshape((self.n_joints, 1)) * self.dof_velocity_scale,
            self.prev_action
        ]).reshape((1, -1))

        # run policy
        action = run_inference(self.model, obs, det=True).flatten()
        self.prev_action = action.reshape((self.n_joints, 1))

        # hold default pose at startup
        now = self.get_clock().now()
        if (now - self.start_time).nanoseconds * 1e-9 < self._warmup_duration:
            target = self.default_pose
        else:
            target = action * self.action_scale.flatten() + self.default_pose

        # publish joint targets
        if self.simulation:
            msg = JointState()
        else:
            msg = JointsCommand()
        msg.header.stamp = now.to_msg()
        msg.name = self.joint_names
        msg.position = target.tolist()
        if not self.simulation:
            msg.kp_scale = [1.0]*self.n_joints
            msg.kd_scale = [1.0]*self.n_joints
        self.joint_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = InferenceController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
