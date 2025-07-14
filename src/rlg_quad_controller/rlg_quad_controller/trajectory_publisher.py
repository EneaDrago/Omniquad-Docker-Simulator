import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from pi3hat_moteus_int_msgs.msg import JointsCommand


class TrajectoryPublisher(Node):

        def __init__(self):
               
               super().__init__('trajectory_publisher')

               self.joint_names = ['LF_HFE', 'LH_HFE', 'RF_HFE', 'RH_HFE', 'LF_KFE', 'LH_KFE', 'RF_KFE', 'RH_KFE']
               self.num_joints = len(self.joint_names)
               
               source_topic = '/PD_control/command'
               # source_topic = '/joint_states'
               self.joint_target_pos_sub  = self.create_subscription(JointState, source_topic, self.joint_target_pos_callback, 10)
               self.joint_target_pos_pub = self.create_publisher(JointsCommand, '/joint_controller/command', 10)
               
               self.joint_cmd_msg = JointsCommand()
               self.joint_cmd_msg.name = self.joint_names
               self.joint_cmd_msg.position = [0.] * self.num_joints
               self.joint_cmd_msg.velocity = [0.] * self.num_joints
               self.joint_cmd_msg.effort = [0.] * self.num_joints

               self.joint_cmd_msg.kp_scale = [1.] * self.num_joints
               self.joint_cmd_msg.kd_scale = [1.] * self.num_joints

        def joint_target_pos_callback(self, msg):
                
                self.joint_cmd_msg.header.stamp = rclpy.clock.Clock().now().to_msg()
                for i, joint_name in enumerate(self.joint_names):
                        self.joint_cmd_msg.position[i] = msg.position[msg.name.index(joint_name)]
                self.joint_target_pos_pub.publish(self.joint_cmd_msg)


def main(args=None):
    rclpy.init(args=args)
    trajectory_publisher = TrajectoryPublisher()
    rclpy.spin(trajectory_publisher)
    trajectory_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
