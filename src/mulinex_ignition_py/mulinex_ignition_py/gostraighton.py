import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray

class GoStraightOn(Node):

    def __init__(self):
        super().__init__('gostraighton')

        # Publisher 1: per il controllo posizione delle gambe
        self.leg_pub = self.create_publisher(JointState, '/pd_controller/command', 10)

        # Publisher 2: per il controllo velocità delle ruote
        self.wheel_pub = self.create_publisher(Float64MultiArray, '/wheel_velocity_controller/commands', 10)

        self.joint_names = [
            "LF_HFE", "LH_HFE", "RF_HFE", "RH_HFE",     # anche
            "LF_KFE", "LH_KFE", "RF_KFE", "RH_KFE",     # ginocchia
        ]

        hip = math.pi * 120.0 / 180.0
        knee = math.pi * 60.0 / 180.0

        # Posizioni fisse per HFE/KFE
        self.fixed_positions = [
            hip,   -knee,            # Left HIP front, Left KNEE front
           -hip,    knee,            # Left HIP back, Left KNEE back
           -hip,   knee,             # Right HIP front, Right KNEE front
            hip,  -knee,             # Right HIP back, Right KNEE back
        ]

        self.wheel_speed = 500.0  # rad/s

        # Timer per pubblicare comandi a 50 Hz
        self.timer_leg = self.create_timer(0.02, self.send_command_leg)
        self.timer_wheel = self.create_timer(0.02, self.send_command_wheel)

        self.get_logger().info("✅ Nodo 'gostraighton' avviato")

    def send_command_leg(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = self.fixed_positions
        msg.velocity = [0.0] * len(self.fixed_positions)
        msg.effort = [0.0] * len(self.fixed_positions)

        self.leg_pub.publish(msg)

    def send_command_wheel(self):
        msg = Float64MultiArray()
        msg.data = [self.wheel_speed] * 4  # LF, LH, RF, RH
        self.wheel_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = GoStraightOn()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
