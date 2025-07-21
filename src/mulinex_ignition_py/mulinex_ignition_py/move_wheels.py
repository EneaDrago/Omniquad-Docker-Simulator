import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool
from pi3hat_moteus_int_msgs.msg import OmniMulinexCommand
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.duration import Duration
from rclpy.time import Time


''' This node moves the robot using the omni_velocity controller.
This controller is used to control the robot's movement in a 2D plane. 
It takes linear and angular velocities as input, solves the inverse kinematics, and sends commands to the motors.'''

class OmniVelCommander(Node):
    def __init__(self):
        super().__init__('omni_vel_commander')

        # Declare parameters with default values
        self.declare_parameter('vx', 0.5)
        self.declare_parameter('vy', -0.0)
        self.declare_parameter('w', -0.5)
        self.declare_parameter('publish_rate', 10)

        # Get parameter values
        self.vx = self.get_parameter('vx').get_parameter_value().double_value
        self.vy = self.get_parameter('vy').get_parameter_value().double_value
        self.w  = self.get_parameter('w').get_parameter_value().double_value
        self.rate_hz = self.get_parameter('publish_rate').get_parameter_value().integer_value

        # QoS profile
        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            #deadline=QoSDuration(nanoseconds=int(1e9 / self.rate_hz + 5e6))
            deadline = Duration(seconds=0, nanoseconds=int(1e9 / self.rate_hz + 5e6))

        )
        
        # Publisher
        self.publisher = self.create_publisher(OmniMulinexCommand, '/omni_control/command', qos)

        # Homing service
        self.homing_client = self.create_client(SetBool, '/omni_control/homing_srv')
        if not self.homing_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error('Homing service not available')
            rclpy.shutdown()
            return

        req = SetBool.Request()
        req.data = True

        future = self.homing_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        if not future.result() or not future.result().success:
            self.get_logger().error('Homing failed')
            rclpy.shutdown()
            return

        self.get_logger().info('Homing succeeded—controller ACTIVE')

        # Timer and time tracking
        self.start_time = self.get_clock().now()
        self.timeout_duration = Duration(seconds=10.0)

        self.timer = self.create_timer(1.0 / self.rate_hz, self.publish_cmd)

    def publish_cmd(self):
        now = self.get_clock().now()
        timeout = (now - self.start_time) > self.timeout_duration

        msg = OmniMulinexCommand()
        msg.v_x = 0.0 if timeout else self.vx
        msg.v_y = 0.0 if timeout else self.vy
        msg.omega = 0.0 if timeout else self.w
        msg.height_rate = 0.0
        msg.header.stamp = now.to_msg()

        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = OmniVelCommander()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
