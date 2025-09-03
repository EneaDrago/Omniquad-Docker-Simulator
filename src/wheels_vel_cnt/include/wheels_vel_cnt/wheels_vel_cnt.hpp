#ifndef WHEELS_VEL_CNT_HPP_
#define WHEELS_VEL_CNT_HPP_

#include <string>
#include <vector>
#include <mutex>

#include "controller_interface/controller_interface.hpp"
#include "rclcpp/subscription.hpp"
#include "custom_interfaces/msg/wheel_velocity_command.hpp"
#include "hardware_interface/loaned_state_interface.hpp"
#include "sensor_msgs/msg/joint_state.hpp"

namespace wheels_vel_cnt
{
class WheelsVelCnt : public controller_interface::ControllerInterface
{
public:
  controller_interface::InterfaceConfiguration command_interface_configuration() const override;
  controller_interface::InterfaceConfiguration state_interface_configuration() const override;
  CallbackReturn on_init() override;
  CallbackReturn on_configure(const rclcpp_lifecycle::State &previous_state) override;
  CallbackReturn on_activate(const rclcpp_lifecycle::State &previous_state) override;
  CallbackReturn on_deactivate(const rclcpp_lifecycle::State &previous_state) override;
  controller_interface::return_type update(const rclcpp::Time &time, const rclcpp::Duration &period) override;

private:
  void cmd_callback(const custom_interfaces::msg::WheelVelocityCommand::SharedPtr msg);

  std::vector<std::string> wheel_names_ = {"RF_WHEEL_JNT", "LF_WHEEL_JNT", "RH_WHEEL_JNT", "LH_WHEEL_JNT"};
  std::array<double, 4> last_command_ = {0.0, 0.0, 0.0, 0.0};
  std::mutex command_mutex_;
  rclcpp::Subscription<custom_interfaces::msg::WheelVelocityCommand>::SharedPtr cmd_sub_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr wheel_state_pub_;

  //std::vector<std::string> joint_names_;
  std::vector<double> measured_velocities_;
  std::vector<double> measured_positions_;
};

} // namespace wheels_vel_cnt

#endif // WHEELS_VEL_CNT_HPP_
