import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist

class GoStraightOn(Node):

    def __init__(self):
        super().__init__('gostraighton')

        # Publisher per il controllo delle gambe (posizione)
        self.leg_pub = self.create_publisher(JointState, '/pd_controller/command', 10)

        # Publisher per il controllo delle ruote (velocità)
        self.wheel_pub = self.create_publisher(Twist, '/omni_control/cmd_vel', 10)

        # Nomi dei giunti delle gambe
        self.joint_names = [
            "LF_HFE", "LH_HFE", "RF_HFE", "RH_HFE",     # anche
            "LF_KFE", "LH_KFE", "RF_KFE", "RH_KFE",     # ginocchia
        ]

        # Posizione target per ciascun giunto
        hip = math.radians(120)
        knee = math.radians(60)

        self.fixed_positions = [
            hip,   -knee,   # LF
           -hip,    knee,   # LH
           -hip,    knee,   # RF
            hip,   -knee    # RH
        ]

        # Velocità desiderata in avanti (realistica)
        self.forward_speed = 0.3  # m/s

        # Timer per pubblicare le posizioni delle gambe (50 Hz)
        self.timer_leg = self.create_timer(0.02, self.send_command_leg)

        # Timer per pubblicare velocità alle ruote (10 Hz è sufficiente)
        self.timer_wheel = self.create_timer(0.1, self.send_command_wheel)

        self.get_logger().info("✅ Nodo 'gostraighton' avviato e in funzione")

    def send_command_leg(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = self.fixed_positions

        self.leg_pub.publish(msg)

    def send_command_wheel(self):
        msg = Twist()
        msg.linear.x = self.forward_speed
        msg.linear.y = 0.0
        msg.angular.z = 0.0

        self.wheel_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = GoStraightOn()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
