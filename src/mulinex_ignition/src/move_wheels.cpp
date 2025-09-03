#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "std_srvs/srv/set_bool.hpp"
#include "pi3hat_moteus_int_msgs/msg/omni_mulinex_command.hpp"
 
using namespace std::chrono_literals;
 
int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("omni_vel_commander");
 
  // parameters
  node->declare_parameter<double>("vx", 0.5);
  node->declare_parameter<double>("vy", 0.0);
  node->declare_parameter<double>("w", 0.0);
  node->declare_parameter<int>("publish_rate", 10);
  double vx = node->get_parameter("vx").as_double();
  double vy = node->get_parameter("vy").as_double();
  double w  = node->get_parameter("w").as_double();
  int rate_hz = node->get_parameter("publish_rate").as_int();
 
  // QoS (match BestEffort + deadline)
  rclcpp::QoS qos(10);
  qos.reliability(rclcpp::ReliabilityPolicy::BestEffort);
  qos.deadline(std::chrono::milliseconds(1000 / rate_hz + 5));
 
  // 1) publish the correct message type
  using CmdMsg = pi3hat_moteus_int_msgs::msg::OmniMulinexCommand;
  // 2) on the same topic the joystick uses
  auto pub = node->create_publisher<CmdMsg>("/omni_control/command", qos);
 
  // --- HOMING (must set data=true) ---
  using HomingSvc = std_srvs::srv::SetBool;
  auto homing_cli =
    node->create_client<HomingSvc>("/omni_control/homing_srv");
  if (!homing_cli->wait_for_service(2s)) {
    RCLCPP_ERROR(node->get_logger(), "Homing service not available");
    return 1;
  }
  auto homing_req = std::make_shared<HomingSvc::Request>();
  homing_req->data = true;
  auto homing_fut = homing_cli->async_send_request(homing_req);
  if (rclcpp::spin_until_future_complete(node, homing_fut, 2s)
        != rclcpp::FutureReturnCode::SUCCESS
      || !homing_fut.get()->success)
  {
    RCLCPP_ERROR(node->get_logger(), "Homing failed");
    return 1;
  }
  RCLCPP_INFO(node->get_logger(), "Homing succeeded—controller ACTIVE");
 
  // record start time to enforce timeout
  rclcpp::Time start_time = node->now();
  rclcpp::Duration timeout_duration(5s);
 
  // publish loop
  rclcpp::Rate loop_rate(rate_hz);
  while (rclcpp::ok()) {
    auto msg = CmdMsg();
    // after 5 seconds, override velocities to zero
    bool timeout = (node->now() - start_time) > timeout_duration;
    msg.set__v_x(timeout ? 0.0 : vx);
    msg.set__v_y(timeout ? 0.0 : vy);
    msg.set__omega(timeout ? 0.0 : w);
    msg.set__height_rate(0.0);                      // joystick always sets this
    msg.header.stamp = node->now();                 // stamped header
 
    pub->publish(msg);
    rclcpp::spin_some(node);
    loop_rate.sleep();
  }
 
  rclcpp::shutdown();
  return 0;
}