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

class InferenceController(Node):
    def __init__(self):
        super().__init__('inference_controller')

        # Simulation flag
        self.declare_parameter('simulation', False)
        self.simulation = self.get_parameter('simulation').get_parameter_value().bool_value

        # Model path as pth file
        self.declare_parameter('model_path', '')
        self.declare_parameter('config_path', '')
        self.model_path     = self.get_parameter('model_path').get_parameter_value().string_value
        self.config_path    = self.get_parameter('config_path').get_parameter_value().string_value

        # Topic names
        self.declare_parameter('joint_state_topic', '/state_broadcaster/joint_states')
        self.declare_parameter('joint_target_pos_topic', '/joint_controller/command')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.joint_state_topic          = self.get_parameter('joint_state_topic').get_parameter_value().string_value
        self.joint_target_pos_topic     = self.get_parameter('joint_target_pos_topic').get_parameter_value().string_value
        self.cmd_vel_topic              = self.get_parameter('cmd_vel_topic').get_parameter_value().string_value

        # Pub Control rate in [Hz]
        # self.declare_parameter('self.control_pub_rate', 250)
        self.RESAMPLING = False
        self.control_pub_rate = 500
        # Inference rate
        with open(self.config_path, 'r') as f:
            params = yaml.safe_load(f)
        
        # self.declare_parameter('rate', 100)
        # self.rate = self.get_parameter('rate').get_parameter_value().integer_value
        self.rate = 1.0 / (params['task']['sim']['dt'] * params['task']['env']['control']['decimation'])
        rclpy.logging.get_logger('rclpy.node').info('Inference rate: {}'.format(self.rate))

        self.action_scale       = params['task']['env']['control']['actionScale']    # 0.5  
        self.dofPositionScale   = params['task']['env']['learn']['dofPositionScale'] # 1.0
        self.dofVelocityScale   = params['task']['env']['learn']['dofVelocityScale'] # 0.05

        self.linearVelocityScale    = params['task']['env']['learn']['linearVelocityScale'] # 2.0
        self.angularVelocityScale   = params['task']['env']['learn']['angularVelocityScale'] #0.25
        self.cmd_vel_scale  = np.array([self.linearVelocityScale, self.angularVelocityScale]).reshape((2,1))
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
        ])

        self._avg_default_dof  = self.default_dof.tolist()
        # Initialize joint publisher/subscriber
        self.njoint = 8

        self.joint_names=(
            'LF_HFE',   
            'LH_HFE',   
            'RF_HFE',
            'RH_HFE',
            'LF_KFE',   
            'LH_KFE',   
            'RF_KFE',
            'RH_KFE',
        )

        
        self.joint_kp = np.array([
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0])
        


        self.joint_kd = np.array([
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
        
        self.base_vel = np.zeros((3,1))
        self.cmd_vel = np.zeros((2,1)) # speed and heading
        self.base_quat = np.zeros((4,1))
        self.projected_gravity = np.zeros((3,1))
        self.yawboia = 0.0

        # Load PyTorch model and create timer
        rclpy.logging.get_logger('rclpy.node').info('Loading model from {}'.format(self.model_path))
        self.model = build_rlg_model(self.model_path, params)
        self.startup_time = rclpy.clock.Clock().now()
        # start inference
        self.timer = self.create_timer(1.0 / self.rate, self.inference_callback)
        # pub timer
        self.counter = 0
        if self.RESAMPLING:
            self.pub_timer = self.create_timer(1.0 / self.control_pub_rate, self.pub_callback)
        rclpy.logging.get_logger('rclpy.node').info('Model loaded. Node ready for inference.') 


    def pub_callback(self):
        joint_msg = JointsCommand()
        if self.simulation:
            joint_msg = JointState()
        joint_msg.header.stamp = rclpy.clock.Clock().now().to_msg()
        joint_msg.name = self.joint_names

        if rclpy.clock.Clock().now() < (self.startup_time + rclpy.duration.Duration(seconds=3.0)):
            joint_msg.position = self._avg_default_dof
        else:
            for i in range(self.njoint):
                joint_msg.position.append(self.joint_ref_resampled[i][self.counter].tolist())
            #rclpy.logging.get_logger('rclpy.node').info('cmd_pub_res: {}'.format(joint_msg.position))
            rclpy.logging.get_logger('rclpy.node').info('counter: {}'.format(self.counter))
            self.counter += 1
            if self.counter > int(self.control_pub_rate/self.rate) - 1:
                self.counter = int(self.control_pub_rate/self.rate) - 1
            
            
        if not self.simulation:
            joint_msg.kp_scale = self.joint_kp.tolist()
            joint_msg.kd_scale = self.joint_kd.tolist()
        joint_msg.velocity = np.zeros(self.njoint).tolist()
        joint_msg.effort = np.zeros(self.njoint).tolist()
        self.joint_target_pos_pub.publish(joint_msg)

    def cmd_vel_callback(self, msg):
        self.cmd_vel = np.array([msg.linear.x, msg.angular.z]).reshape((2,1))
        np.clip(self.cmd_vel, self.cmd_vel_min, self.cmd_vel_max)


    def imu_callback(self, msg):
        self.base_vel = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]).reshape((3,1))*0.0
        orientation_list = [msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z]
        # rclpy.logging.get_logger('rclpy.node').info('QUAT: {}'.format(orientation_list))
        yaw, pitch, roll = t3d.euler.quat2euler(orientation_list,axes='szyx')
        # rclpy.logging.get_logger('rclpy.node').info('RPY: {}'.format([yaw,pitch,roll]))
        (w,x,y,z) =  t3d.euler.euler2quat(self.yawboia, pitch, roll, axes='szyx')
        # rclpy.logging.get_logger('rclpy.node').info('CORRECTED_QUAT: {}'.format((w,x,y,z)))
        self.base_quat = np.array([w,x,y,z]).reshape((4,1))
        # self.base_quat = np.array([1.0, 0.0, 0.0, 0.0]).reshape((4,1))
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
        
        obs_list = np.concatenate((
            self.projected_gravity,
            self.base_quat,
            self.base_vel * self.angularVelocityScale,
            (self.cmd_vel * self.cmd_vel_scale).reshape((2,1)), 
            np.fromiter(self.joint_pos.values(),dtype=float).reshape((self.njoint,1)) *
                self.dofPositionScale,
            np.fromiter(self.joint_vel.values(),dtype=float).reshape((self.njoint,1)) *
                self.dofVelocityScale,
            self.previous_action, 
        )).reshape((1,36))
        # rclpy.logging.get_logger('rclpy.node').info('Observation vector: {}'.format(obs_list))
        
        # try:
        action = run_inference(self.model, obs_list, det=True)
        joint_ref_new = (np.squeeze(action) * self.action_scale + self.default_dof).tolist()
        joint_ref_prev = (np.squeeze(self.previous_action) * self.action_scale + self.default_dof).tolist()

        self.resampling_reference(joint_ref_prev, joint_ref_new)
        self.counter = 0
        self.previous_action = np.reshape(action,(self.njoint,1))
        
        if not self.RESAMPLING:
            joint_msg = JointsCommand()
            if self.simulation:
                joint_msg = JointState()
            joint_msg.header.stamp = rclpy.clock.Clock().now().to_msg()
            joint_msg.name = self.joint_names
            
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


    def resampling_reference(self, joint_ref_prev, joint_ref_new):
        """
        Resample the reference joint positions from 2 to 5 samples
        """
        joint_ref_prev = np.array(joint_ref_prev).reshape((self.njoint,1))
        joint_ref_new = np.array(joint_ref_new).reshape((self.njoint,1))
        joint_ref = np.concatenate((joint_ref_prev, joint_ref_new), axis=1)
        
        new_timeline = np.linspace(0,1,int(self.control_pub_rate/self.rate))
        old_timeline = np.linspace(0,1,2)
        # rclpy.logging.get_logger('rclpy.node').info('old_timeline: {}'.format(old_timeline))
        self.joint_ref_resampled = np.empty([self.njoint,int(self.control_pub_rate/self.rate)])
        
        for i in range(self.njoint):
            # self.joint_ref_resampled.append(np.interp(new_timeline,old_timeline,joint_ref[i,:]))
            self.joint_ref_resampled[i] = np.interp(new_timeline,old_timeline,joint_ref[i])
        # rclpy.logging.get_logger('rclpy.node').info('joint_ref: {}'.format(self.joint_ref_resampled))


def main(args=None):
    rclpy.init(args=args)
    inference_controller = InferenceController()
    rclpy.spin(inference_controller)
    inference_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
