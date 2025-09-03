import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
from rclpy.qos import QoSProfile, ReliabilityPolicy, QoSDurabilityPolicy

from std_srvs.srv import SetBool
from custom_interfaces.msg import WheelVelocityCommand 

''' This node moves the robot using the wheels_vel_cnt controller.
This controller is used to control the wheels directly, without using the omni_velocity controller.'''


class WheelsVelCommander(Node):
    def __init__(self):
        super().__init__('wheels_vel_commander')

        # Parametri
        self.declare_parameter('v_rf', 10.0)
        self.declare_parameter('v_lf', -10.0)
        self.declare_parameter('v_rb', -10.0)
        self.declare_parameter('v_lb', 10.0)
        self.declare_parameter('publish_rate', 10)

        self.v_rf = self.get_parameter('v_rf').get_parameter_value().double_value
        self.v_lf = self.get_parameter('v_lf').get_parameter_value().double_value
        self.v_rb = self.get_parameter('v_rb').get_parameter_value().double_value
        self.v_lb = self.get_parameter('v_lb').get_parameter_value().double_value
        self.rate_hz = self.get_parameter('publish_rate').get_parameter_value().integer_value

        # qos = QoSProfile(
        #     depth=10,
        #     reliability=ReliabilityPolicy.BEST_EFFORT,
        #     deadline=Duration(nanoseconds=int(1e9 / self.rate_hz + 5e6))
        # )
        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE    
        )

        # Publisher
        self.publisher = self.create_publisher(WheelVelocityCommand, '/wheels_vel_controller/wheels_velocity_cmd', qos)

        # Homing opzionale
        self.homing_client = self.create_client(SetBool, '/wheels_vel_cnt/homing_srv')
        if self.homing_client.wait_for_service(timeout_sec=2.0):
            req = SetBool.Request()
            req.data = True
            future = self.homing_client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
            if not future.result() or not future.result().success:
                self.get_logger().warn('Homing failed')
            else:
                self.get_logger().info('Homing succeeded')

        # Timer
        self.start_time = self.get_clock().now()
        self.timeout_duration = Duration(seconds=10.0)
        self.timer = self.create_timer(1.0 / self.rate_hz, self.publish_cmd)

    def publish_cmd(self):
        now = self.get_clock().now()
        timeout = (now - self.start_time) > self.timeout_duration

        msg = WheelVelocityCommand()
        msg.v_rf = 0.0 if timeout else self.v_rf
        msg.v_lf = 0.0 if timeout else self.v_lf
        msg.v_rb = 0.0 if timeout else self.v_rb
        msg.v_lb = 0.0 if timeout else self.v_lb

        self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = WheelsVelCommander()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


