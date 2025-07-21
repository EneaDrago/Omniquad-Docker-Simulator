#include "wheels_vel_cnt/wheels_vel_cnt.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "controller_interface/helpers.hpp"

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
  wheel_state_pub_ = get_node()->create_publisher<sensor_msgs::msg::JointState>("/Wheel_Feedback", 10);
  return CallbackReturn::SUCCESS;
}

CallbackReturn WheelsVelCnt::on_activate(const rclcpp_lifecycle::State &)
{
  // Nessun push_back! Solo verifica che le interfacce siano disponibili e ordinabili
  std::vector<std::reference_wrapper<hardware_interface::LoanedStateInterface>> tmp_vel, tmp_pos;

  if (
    !controller_interface::get_ordered_interfaces(
      state_interfaces_, wheel_names_, "velocity", tmp_vel) ||
    !controller_interface::get_ordered_interfaces(
      state_interfaces_, wheel_names_, "position", tmp_pos))
  {
    RCLCPP_ERROR(get_node()->get_logger(), "Failed to get ordered state interfaces (velocity/position)");
    return CallbackReturn::ERROR;
  }

  measured_positions_.resize(wheel_names_.size(), 0.0);
  measured_velocities_.resize(wheel_names_.size(), 0.0);

  RCLCPP_INFO(get_node()->get_logger(), "WheelsVelCnt controller activated successfully.");
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

// controller_interface::InterfaceConfiguration WheelsVelCnt::state_interface_configuration() const
// {
//   return controller_interface::InterfaceConfiguration{
//     controller_interface::interface_configuration_type::NONE, {}};
// }
controller_interface::InterfaceConfiguration WheelsVelCnt::state_interface_configuration() const
{
	controller_interface::InterfaceConfiguration config;
	config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
	for (const auto &wheel_name : wheel_names_)
	{
		config.names.push_back(wheel_name + "/velocity");
		config.names.push_back(wheel_name + "/position");
	}
	return config;
}

controller_interface::return_type WheelsVelCnt::update(const rclcpp::Time &, const rclcpp::Duration &)
{
  std::vector<std::reference_wrapper<hardware_interface::LoanedStateInterface>> tmp_vel, tmp_pos;

  if (
    !controller_interface::get_ordered_interfaces(
      state_interfaces_, wheel_names_, "velocity", tmp_vel) ||
    !controller_interface::get_ordered_interfaces(
      state_interfaces_, wheel_names_, "position", tmp_pos))
  {
    RCLCPP_ERROR(get_node()->get_logger(), "Failed to get state interfaces during update()");
    return controller_interface::return_type::ERROR;
  }

  for (size_t i = 0; i < wheel_names_.size(); ++i)
  {
    measured_velocities_[i] = tmp_vel[i].get().get_value();
    measured_positions_[i]  = tmp_pos[i].get().get_value();

    RCLCPP_DEBUG(get_node()->get_logger(), "Wheel %s: vel=%.3f, pos=%.3f",
      wheel_names_[i].c_str(), measured_velocities_[i], measured_positions_[i]);
  }

  // Pubblica sul topic /Wheel_Feedback
  sensor_msgs::msg::JointState msg;
  msg.header.stamp = get_node()->now();
  msg.name = wheel_names_;
  msg.position = measured_positions_;
  msg.velocity = measured_velocities_;
  wheel_state_pub_->publish(msg);

  // Applica i comandi
  std::lock_guard<std::mutex> lock(command_mutex_);
  for (size_t i = 0; i < wheel_names_.size(); ++i)
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
