#include "wheels_vel_cnt/wheels_vel_cnt.hpp"
#include "pluginlib/class_list_macros.hpp"

namespace wheels_vel_cnt
{

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

CallbackReturn WheelsVelCnt::on_init()
{
  return CallbackReturn::SUCCESS;
}

CallbackReturn WheelsVelCnt::on_configure(const rclcpp_lifecycle::State &)
{
  cmd_sub_ = get_node()->create_subscription<custom_interfaces::msg::WheelVelocityCommand>(
    "~/wheels_velocity_cmd", 10,
    std::bind(&WheelsVelCnt::cmd_callback, this, std::placeholders::_1));
  return CallbackReturn::SUCCESS;
}

CallbackReturn WheelsVelCnt::on_activate(const rclcpp_lifecycle::State &)
{
  return CallbackReturn::SUCCESS;
}

CallbackReturn WheelsVelCnt::on_deactivate(const rclcpp_lifecycle::State &)
{
  return CallbackReturn::SUCCESS;
}

controller_interface::InterfaceConfiguration WheelsVelCnt::command_interface_configuration() const
{
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (const auto &name : wheel_names_)
    config.names.push_back(name + "/velocity");
  return config;
}

controller_interface::InterfaceConfiguration WheelsVelCnt::state_interface_configuration() const
{
  return controller_interface::InterfaceConfiguration{
    controller_interface::interface_configuration_type::NONE, {}};
}

controller_interface::return_type WheelsVelCnt::update(const rclcpp::Time &, const rclcpp::Duration &)
{
  std::lock_guard<std::mutex> lock(command_mutex_);
  for (size_t i = 0; i < 4; ++i)
    command_interfaces_[i].set_value(last_command_[i]);
  return controller_interface::return_type::OK;
}

void WheelsVelCnt::cmd_callback(const custom_interfaces::msg::WheelVelocityCommand::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(command_mutex_);
  last_command_ = {msg->v_rf, msg->v_lf, msg->v_rb, msg->v_lb};
  RCLCPP_INFO(get_node()->get_logger(), "Received cmd: rf=%.2f, lf=%.2f, rb=%.2f, lb=%.2f",
              msg->v_rf, msg->v_lf, msg->v_rb, msg->v_lb);
}

} // namespace wheels_vel_cnt

PLUGINLIB_EXPORT_CLASS(wheels_vel_cnt::WheelsVelCnt, controller_interface::ControllerInterface)
