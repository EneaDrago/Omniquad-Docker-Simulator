import rclpy
import torch
import numpy as np
import yaml
from rclpy.node import Node
from pi3hat_moteus_int_msgs.msg import JointsCommand, JointsStates, PacketPass
from sensor_msgs.msg import  Imu, JointState
from geometry_msgs.msg import Twist 
import transforms3d as t3d

from .utils.torch_utils import quat_rotate_inverse, quat_rotate_inverse_numpy
from .utils.rlg_utils import build_rlg_model, run_inference

"""
This node subscribes to the joint states and cmd_vel topics, and publishes the target joint positions.

cmd_vel  ---> | inference_controller | ---> joint_target_pos --> PD contr

"""

# …

class InferenceController(Node):
    def __init__(self):
        super().__init__("inference_controller")

        # ------------------------------------------------------------------
        # nuovi parametri ROS: path agent + env
        # ------------------------------------------------------------------
        self.declare_parameter("model_path", "")
        self.declare_parameter("env_cfg_path", "")
        self.declare_parameter("agent_cfg_path", "")
        self.model_path     = self.get_parameter("model_path").get_parameter_value().string_value
        self.env_cfg_path   = self.get_parameter("env_cfg_path").get_parameter_value().string_value
        self.agent_cfg_path = self.get_parameter("agent_cfg_path").get_parameter_value().string_value

        # ------------------------------------------------------------------
        # leggo gli YAML UNA sola volta, ci serviranno più avanti
        # ------------------------------------------------------------------
        with open(self.env_cfg_path, "r") as f:
            self.env_cfg = yaml.safe_load(f)
        with open(self.agent_cfg_path, "r") as f:
            self.agent_cfg = yaml.safe_load(f)

        # ------------------------------------------------------------------
        # frequenza inferenza  = 1 / (dt * decimation)
        # ------------------------------------------------------------------
        dt          = self.env_cfg["sim"]["dt"]
        decimation  = self.env_cfg["decimation"]
        self.rate   = 1.0 / (dt * decimation)
        self.get_logger().info(f"Inference rate: {self.rate:.2f} Hz")

        # ------------------------------------------------------------------
        # scale azioni / osservazioni (campi mutati nel nuovo env.yaml)
        # ------------------------------------------------------------------
        leg_scale   = self.env_cfg["actions"]["joint_pos"]["scale"]     # 0.1
        wheel_scale = self.env_cfg["actions"]["joint_vel"]["scale"]     # 50
        self.action_scale = np.array([leg_scale]*8 + [wheel_scale]*4).reshape((12,1))

        learn_cfg   = self.env_cfg["observations"]                      # corrisponde alle vecchie metriche
        self.dofPositionScale   = self.agent_cfg["params"]["env"]["clip_actions"]     # esempio
        self.dofVelocityScale   = 0.05                                  # se non c’è lo fissi

        # ------------------------------------------------------------------
        # carico il modello
        # ------------------------------------------------------------------
        self.get_logger().info(f"Loading rl‑games checkpoint '{self.model_path}'…")
        self.model = build_rlg_model(
            self.model_path,
            self.env_cfg_path,
            self.agent_cfg_path,
            device="cuda:0"
        )


        with open(self.config_path, 'r') as f:
            params = yaml.safe_load(f)

        self.linearVelocityScale    = params['task']['env']['learn']['linearVelocityScale'] # 2.0
        self.angularVelocityScale   = params['task']['env']['learn']['angularVelocityScale'] #0.25
        self.cmd_vel_scale  = np.array([self.linearVelocityScale, self.linearVelocityScale, self.angularVelocityScale]).reshape((3,1))
        self.cmd_vel_min    = np.array([-1.0, -1.0]).reshape((2,1))
        self.cmd_vel_max    = np.array([1.0, 1.0]).reshape((2,1))
        
        self.hip_angle = 120.0
        self.knee_angle = 60.0

        self.default_dof    = np.array([
                np.deg2rad(self.hip_angle),     
                -(np.deg2rad(self.hip_angle)),    
                -(np.deg2rad(self.hip_angle)),
                np.deg2rad(self.hip_angle),
                -np.deg2rad(self.knee_angle),   
                np.deg2rad(self.knee_angle),    
                np.deg2rad(self.knee_angle),
                -np.deg2rad(self.knee_angle),
                0.0,
                0.0,
                0.0,
                0.0
        ])

        self._avg_default_dof  = self.default_dof.tolist()
        # Initialize joint publisher/subscriber
        self.njoint = 12

        self.joint_names=(
            'LF_HFE',   
            'LH_HFE',   
            'RF_HFE',
            'RH_HFE',
            'LF_KFE',   
            'LH_KFE',   
            'RF_KFE',
            'RH_KFE',
            'LF_WHEEL_JNT',
            'LH_WHEEL_JNT',
            'RF_WHEEL_JNT',
            'RH_WHEEL_JNT',
        )

        
        self.joint_kp = np.array([
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0])
        


        self.joint_kd = np.array([
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0])

            
        if self.simulation:
            self.joint_target_pos_pub = self.create_publisher(JointState, self.joint_target_pos_topic, 10)
            self.joint_sub  = self.create_subscription(JointState, self.joint_state_topic, self.joint_state_callback, 10)
        else:
            self.joint_target_pos_pub = self.create_publisher(JointsCommand, self.joint_target_pos_topic, 10)
            self.joint_sub  = self.create_subscription(JointsStates, self.joint_state_topic, self.joint_state_callback, 10)

        self.cmd_sub    = self.create_subscription(Twist, self.cmd_vel_topic, self.cmd_vel_callback, 10)

        self.declare_parameter('use_imu', False)
        self.declare_parameter('imu_topic', '/IMU_Broadcaster/imu')
        self.use_imu = self.get_parameter('use_imu').get_parameter_value().bool_value
        if self.use_imu:
            self.imu_topic = self.get_parameter('imu_topic').get_parameter_value().string_value
            self.imu_sub = self.create_subscription(Imu, self.imu_topic, self.imu_callback, 10)

        # Initialize buffers as dicts, so it's independent of the order of the joints
        self.joint_pos = {self.joint_names[i]:0.0 for i in range(self.njoint)}
        self.joint_vel = {self.joint_names[i]:0.0 for i in range(self.njoint)}
        self.previous_joint_pos =self.joint_pos.copy()
        self.prev_timestamp = 0.0

        self.previous_action = np.zeros((self.njoint,1))
        
        self.base_ang_vel = np.zeros((3,1))
        self.cmd_vel = np.zeros((3,1)) # speed and heading
        self.base_quat = np.zeros((4,1))
        self.projected_gravity = np.zeros((3,1))
        self.yawboia = 0.0

        # Load PyTorch model and create timer
        rclpy.logging.get_logger('rclpy.node').info('Loading model from {}'.format(self.model_path))
        self.model = build_rlg_model(self.model_path, params)
        self.startup_time = rclpy.clock.Clock().now()
        # start inference
        self.timer = self.create_timer(1.0 / self.rate, self.inference_callback)
        rclpy.logging.get_logger('rclpy.node').info('Model loaded. Node ready for inference.') 


    def cmd_vel_callback(self, msg):
        self.cmd_vel = np.array([msg.linear.x, msg.linear.y, msg.angular.z]).reshape((3,1))
        np.clip(self.cmd_vel, self.cmd_vel_min, self.cmd_vel_max)


    def imu_callback(self, msg):
        self.base_ang_vel = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]).reshape((3,1))
        orientation_list = [msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z]
        # rclpy.logging.get_logger('rclpy.node').info('QUAT: {}'.format(orientation_list))
        yaw, pitch, roll = t3d.euler.quat2euler(orientation_list,axes='szyx')
        # rclpy.logging.get_logger('rclpy.node').info('RPY: {}'.format([yaw,pitch,roll]))
        (w,x,y,z) =  t3d.euler.euler2quat(self.yawboia, pitch, roll, axes='szyx')
        # rclpy.logging.get_logger('rclpy.node').info('CORRECTED_QUAT: {}'.format((w,x,y,z)))
        self.base_quat = np.array([w,x,y,z]).reshape((4,1))
        # UNCOMMENT TO LOAD QUAT FROM IMU (USE WITH CAUTION, THERE IS DRIFT ON THE YAW)
        # self.base_quat = np.array([msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z]).reshape((4,1))
        self.projected_gravity = quat_rotate_inverse_numpy(self.base_quat, np.array([0,0,-1.0]).reshape((3,1)))


    def joint_state_callback(self, msg):
        # Assign to dict using the names in msg.name
        t = rclpy.clock.Clock().now()
        timestamp = t.nanoseconds / 1e9 # [s]
        # rclpy.logging.get_logger('rclpy.node').info('{}'.format(timestamp - self.prev_timestamp))
        for i in range(self.njoint):
            if (not np.isnan(msg.position[i]) and (not np.isnan(msg.velocity[i]))):
                self.joint_pos[msg.name[i]] = msg.position[i]
                self.joint_vel[msg.name[i]] = msg.velocity[i]
            # UNCOMMENT TO COMPUTE VEL BY DERIVATION 
            # self.joint_vel[msg.name[i]] = (msg.position[i] - self.previous_joint_pos[msg.name[i]]) / (timestamp - self.prev_timestamp)
            self.previous_joint_pos[msg.name[i]] = msg.position[i]
        self.prev_timestamp = timestamp


    def warmup_controller(self):
        joint_msg = JointsCommand()
        if self.simulation:
            joint_msg = JointState()
        joint_msg.header.stamp = rclpy.clock.Clock().now().to_msg()
        joint_msg.name = self.joint_names
        joint_msg.position = (self.default_dof).tolist()
        if not self.simulation:
            joint_msg.kp_scale = self.joint_kp.tolist()
            joint_msg.kd_scale = self.joint_kd.tolist()
        joint_msg.velocity = np.zeros(self.njoint).tolist()
        joint_msg.effort = np.zeros(self.njoint).tolist()
        self.joint_target_pos_pub.publish(joint_msg)


    def inference_callback(self):
        """
        Callback function for inference timer. Infers joints target_pos from model and publishes it.
        """
        # +---------------------------------------------------------+
        # | Active Observation Terms in Group: 'policy' (shape: (41,)) |
        # +-----------+---------------------------------+-----------+
        # |   Index   | Name                            |   Shape   |
        # +-----------+---------------------------------+-----------+
        # |     0     | base_ang_vel                    |    (3,)   |
        # |     1     | projected_gravity               |    (3,)   |
        # |     2     | velocity_commands               |    (3,)   |
        # |     3     | joint_pos                       |    (8,)   |
        # |     4     | joint_vel                       |   (12,)   |
        # |     5     | actions                         |   (12,)   |
        # +-----------+---------------------------------+-----------+

        obs_list = np.concatenate((
            self.base_ang_vel * self.angularVelocityScale,
            self.projected_gravity,
            (self.cmd_vel * self.cmd_vel_scale).reshape((3,1)), 
            np.fromiter(self.joint_pos.values(),dtype=float).reshape((self.njoint,1)) *
                self.dofPositionScale,
            np.fromiter(self.joint_vel.values(),dtype=float).reshape((self.njoint,1)) *
                self.dofVelocityScale,
            self.previous_action, 
        )).reshape((1,41))
        # rclpy.logging.get_logger('rclpy.node').info('Observation vector: {}'.format(obs_list))
        
        # try:
        action = run_inference(self.model, obs_list, det=True)
        joint_msg = JointsCommand()
        if self.simulation:
            joint_msg = JointState()
        joint_msg.header.stamp = rclpy.clock.Clock().now().to_msg()
        joint_msg.name = self.joint_names
        self.previous_action = np.reshape(action,(self.njoint,1))
        action = np.squeeze(action)

        if rclpy.clock.Clock().now() < (self.startup_time + rclpy.duration.Duration(seconds=3.0)):
            action *= 0.0
            joint_msg.position = self._avg_default_dof
        else:
            joint_msg.position = (np.squeeze(action) * self.action_scale + self.default_dof).tolist()
            
        if not self.simulation:
            joint_msg.kp_scale = self.joint_kp.tolist()
            joint_msg.kd_scale = self.joint_kd.tolist()
        joint_msg.velocity = np.zeros(self.njoint).tolist()
        joint_msg.effort = np.zeros(self.njoint).tolist()
        self.joint_target_pos_pub.publish(joint_msg)


        

def main(args=None):
    rclpy.init(args=args)
    inference_controller = InferenceController()
    rclpy.spin(inference_controller)
    inference_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
