import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

class GetUpNode(Node):

    def __init__(self):
        super().__init__('getup_node')
        self.sent = False

        self.pub = self.create_publisher(JointState, '/pd_controller/command', 10)

        self.joint_names = [
            "LF_HFE", "LH_HFE", "RF_HFE", "RH_HFE",
            "LF_KFE", "LH_KFE", "RF_KFE", "RH_KFE"
            "LF_ANKLE", "LH_ANKLE", "RF_ANKLE", "RH_ANKLE"
        ]

        hip = math.pi * 120.0 / 180.0
        knee = math.pi * 60.0 / 180.0
        ankle_speed = 100.0
        self.target_positions = [
            hip,   -knee,            # Left HIP front, Left KNEE front
           -hip,    knee,            # Left HIP back, Left KNEE back
           -hip,   knee,             # Right HIP front, Right KNEE front
            hip,  -knee ,            # Right HIP back, Right KNEE back
            0.0, 0.0, 0.0, 0.0,  
        ]
        self.target_velocities = [
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            ankle_speed, -ankle_speed, ankle_speed, ankle_speed 
        ]

        # Timer to wait ~1 second before sending command once
        self.timer = self.create_timer(1.0, self.send_once)

    def send_once(self):
        if self.sent:
            return

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = self.target_positions
        msg.velocity = [0.0] * len(self.target_positions) #100*self.target_velocities  
        msg.effort = [0.0] * len(self.target_positions)

        self.pub.publish(msg)
        self.get_logger().info('✅ get-up JointState command sent')
        self.sent = True
        self.timer.cancel()

def main(args=None):
    rclpy.init(args=args)
    node = GetUpNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

