import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from rclpy.clock import Clock

class WalkingNode(Node):

    def __init__(self):
        super().__init__('walking_node')
        self.pub = self.create_publisher(JointState, '/pd_controller/command', 10)

        self.joint_names = [
            "LF_HFE", "LH_HFE", "RF_HFE", "RH_HFE",
            "LF_KFE", "LH_KFE", "RF_KFE", "RH_KFE",
            "LF_ANKLE", "LH_ANKLE", "RF_ANKLE", "RH_ANKLE"
        ]

        self.timer = self.create_timer(0.02, self.walk)  # 50 Hz
        self.start_time = self.get_clock().now().nanoseconds / 1e9

    def walk(self):
        t = self.get_clock().now().nanoseconds / 1e9 - self.start_time

        hip_amp = math.radians(15)   # ampiezza oscillazione anca
        knee_amp = math.radians(30)  # ampiezza oscillazione ginocchio

        freq = 1.0  # Hz

        # Semplice camminata sinusoidale alternata
        LF_HFE =  hip_amp * math.sin(2 * math.pi * freq * t)
        RF_HFE = -hip_amp * math.sin(2 * math.pi * freq * t)
        LH_HFE = -hip_amp * math.sin(2 * math.pi * freq * t)
        RH_HFE =  hip_amp * math.sin(2 * math.pi * freq * t)

        LF_KFE = -knee_amp * math.sin(2 * math.pi * freq * t)
        RF_KFE =  knee_amp * math.sin(2 * math.pi * freq * t)
        LH_KFE =  knee_amp * math.sin(2 * math.pi * freq * t)
        RH_KFE = -knee_amp * math.sin(2 * math.pi * freq * t)

        # Ankle fissi per semplicità
        ankles = [0.0, 0.0, 0.0, 0.0]

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = [LF_HFE, LH_HFE, RF_HFE, RH_HFE,
                        LF_KFE, LH_KFE, RF_KFE, RH_KFE] + ankles
        msg.velocity = [0.0] * len(msg.position)
        msg.effort = [0.0] * len(msg.position)

        self.pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = WalkingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


