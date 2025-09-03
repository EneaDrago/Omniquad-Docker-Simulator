from typing import List
from rclpy.context import Context
from rclpy.node import Node
from pi3hat_moteus_int_msgs.msg import JointsCommand
from custom_interfaces.msg import WheelVelocityCommand 
from sensor_msgs.msg import JointState
from rclpy.parameter import Parameter
from rclpy.publisher import Publisher
from rclpy.subscription import Subscription
import rclpy
import numpy as np
import time

'''
custom_interfaces.msg > WheelVelocityCommand 

float64 v_rf
float64 v_lf
float64 v_rb
float64 v_lb

Comando da dare al robot

'''

class Bridge_Node(Node):
    def __init__(self):
        super().__init__("bridge")

        # create the publisher 
        self.publisher_ = self.create_publisher(JointsCommand, 'joint_controller/command', 10)
        self.robot_msg = JointsCommand()
        self.sub_wheels  = self.create_subscription(WheelVelocityCommand,'wheels_vel_controller/wheels_velocity_cmd',self.sub_wheels_callback,10)
        self.sub_PD_controller = self.create_subscription(JointsCommand,'/pd_controller/command',self.sub_PD_callback,10)
                        
        self.njoint = 12
        self.joint_vel = [0.0]*self.njoint
        # self.names = ['LF_HFE', 'LF_KFE', 'LH_HFE', 'LH_KFE', 
        #               'RF_HFE', 'RF_KFE', 'RH_HFE', 'RH_KFE',
        #               'LF_WHEEL','LH_WHEEL','RF_WHEEL','RH_WHEEL']
        # self.position_default = [2.094, -1.047, -2.094, 1.047, -2.094, 1.047, 2.094, -1.047, np.nan, np.nan, np.nan, np.nan]
        self.names=['LF_HFE','LH_HFE','RF_HFE','RH_HFE','LF_KFE','LH_KFE','RF_KFE','RH_KFE','LF_WHEEL','LH_WHEEL','RF_WHEEL','RH_WHEEL']
        # self.names=['LF_HFE','LH_HFE','RF_HFE','RH_HFE','LF_KFE','LH_KFE','RF_KFE','RH_KFE']
        self.hip_angle = 120.0
        self.knee_angle = 60.0
        self.position_default    = [
                np.deg2rad(self.hip_angle),     
                -(np.deg2rad(self.hip_angle)),    
                -(np.deg2rad(self.hip_angle)),
                np.deg2rad(self.hip_angle),
                -np.deg2rad(self.knee_angle),   
                np.deg2rad(self.knee_angle),    
                np.deg2rad(self.knee_angle),
                -np.deg2rad(self.knee_angle),
                np.nan, np.nan, np.nan, np.nan
        ]

        self.command_msg = JointsCommand()
        self.command_msg.name = self.names
        self.command_msg.position = self.position_default   
        self.command_msg.effort = np.zeros(self.njoint).tolist()
        self.command_msg.kp_scale = np.ones(self.njoint).tolist()
        self.command_msg.kd_scale = np.ones(self.njoint).tolist()


    def sub_PD_callback(self, msg):
        self.command_msg.header.stamp = rclpy.clock.Clock().now().to_msg()
        self.command_msg.position = msg.position
        # Aggiunge 4 np.nan in fondo per le ruote
        self.command_msg.position.extend([float('nan')] * 4)

        self.publisher_.publish(self.command_msg)
        rclpy.logging.get_logger('rclpy.node').info('PD message converted.') 

    def sub_wheels_callback(self, joint_msg):
        self.joint_vel[8]  = -joint_msg.v_lf
        self.joint_vel[10] = joint_msg.v_rf
        self.joint_vel[9]  = -joint_msg.v_lb
        self.joint_vel[11] = joint_msg.v_rb

        self.command_msg.header.stamp = rclpy.clock.Clock().now().to_msg()
        self.command_msg.velocity = self.joint_vel
        
        self.publisher_.publish(self.command_msg)
        rclpy.logging.get_logger('rclpy.node').info('Wheels message converted') 


def main(args=None):
    rclpy.init(args=args)
    bridge_node = Bridge_Node()
    rclpy.spin(bridge_node)
    bridge_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main