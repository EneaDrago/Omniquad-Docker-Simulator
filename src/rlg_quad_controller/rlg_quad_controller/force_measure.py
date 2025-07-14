import rclpy
import torch
import numpy as np
import yaml
import time
from rclpy.node import Node
from pi3hat_moteus_int_msgs.msg import JointsCommand, JointsStates, PacketPass
from sensor_msgs.msg import  Imu, JointState
from geometry_msgs.msg import Twist 
import transforms3d as t3d


class Forced_Measure(Node):
        def __init__(self):
                super().__init__('force_input')
                self.init_pos = np.array([
                2.094,     # flip
                -2.094,    # flip
                -2.094,
                2.094,
                -1.0472,   # flip
                1.0472,    # flip
                1.0472,
                -1.0472,
                ])

                # EDIT THIS TO SET SIMULATION 
                self.simulation = False
                self.first = True
                self.t = self.get_clock().now()
                self.dt = 0
                self.pos = []
                self.vel = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
                if self.simulation:
                        print("***********************************READING FROM SIMULATION TOPIC **************************")
                        self.sub = self.create_subscription(JointState,'/PD_control/command',self.cmd_sub_clbk,10)
                        self.pub_jnt_stt = self.create_publisher(JointState,'/state_broadcaster/joints_state_forced',10)
                else:
                        self.sub = self.create_subscription(JointsCommand,'/joint_controller/command',self.cmd_sub_clbk,10)
                        self.pub_jnt_stt = self.create_publisher(JointsStates,'/state_broadcaster/joints_state_forced',10)

                self.pub_imu = self.create_publisher(Imu,'/IMU_Broadcaster/imu_forced',10)
                # self.pub_v_cmd = self.create_publisher(Twist,'/cmd_vel',10) # command vel is published by locom_experimes
                # time.sleep(1)
                # self.dt = self.get_clock().now() - self.t
                # print(self.dt.nanoseconds)

        def cmd_sub_clbk(self,msg):
                if(self.first):
                        #get the time and do not pub nothing 
                        self.t = self.get_clock().now()
                        self.pos = msg.position
                        
                        self.first = False

                else:
                        #create the data un pub it 
                        imu_msg = Imu()
                        if self.simulation:
                                msr_msg = JointState()
                                msr_msg.name = ['LF_HFE','LH_HFE','RF_HFE','RH_HFE','LF_KFE','LH_KFE','RF_KFE','RH_KFE']
                        else:
                                msr_msg = JointsStates()
                                msr_msg.name = ['LF_HFE','LH_HFE','RF_HFE','RH_HFE','LF_KFE','LH_KFE','RF_KFE','RH_KFE']
                        # cmd_msg = Twist()
                        # cmd_msg.linear.x = 0.5
                        # cmd_msg.linear.z = 0.0
                        # self.pub_v_cmd.publish(cmd_msg)
                        imu_msg.angular_velocity.x = 0.0
                        imu_msg.angular_velocity.y = 0.0
                        imu_msg.angular_velocity.z = 0.0
                        imu_msg.orientation.w = 1.0
                        imu_msg.orientation.x = 0.0
                        imu_msg.orientation.y = 0.0
                        imu_msg.orientation.z = 0.0
                        self.pub_imu.publish(imu_msg)

                        now = self.get_clock().now()
                        self.dt =( now - self.t ).nanoseconds/pow(10,9)
                        # HARDCODING SELF.DT
                        self.dt = 0.02
                        self.t = now
                        msr_msg.position = msg.position
                        print(self.dt)
                        msr_msg.velocity = self.vel
                        for i in range(8):
                                msr_msg.velocity[i] = (msg.position[i] - self.pos[i])/self.dt
                        self.pos = msg.position
                        self.pub_jnt_stt.publish(msr_msg)






def main(args=None):
    rclpy.init(args=args)
    RL_node = Forced_Measure()
    rclpy.spin(RL_node)
    RL_node.destroy_node()
    rclpy.shutdown()
    


if __name__ == '__main__':
    main()
