#include <chrono>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"

using namespace std::chrono_literals;

class GetUpNode : public rclcpp::Node
{
public:
  GetUpNode()
  : Node("getup_node"), sent_(false)
  {
    // Make sure this matches the controller instance name:
    // e.g. pd_cnt_controller if you did:
    //   ros2 control load_controller --set-state active pd_cnt_controller
    pub_ = this->create_publisher<sensor_msgs::msg::JointState>(
      "/pd_controller/command", 10);

    joint_names_ = {
      "LF_HFE", "LH_HFE", "RF_HFE", "RH_HFE",
      "LF_KFE", "LH_KFE", "RF_KFE", "RH_KFE"
    };

    // “Get up” angles in radians:
    const double hip  = M_PI * 150.0 / 180.0;
    const double knee = M_PI *  60.0  / 180.0;
    target_positions_ = {
      hip,   -hip,
     -hip,    hip,
     -knee,   knee,
      knee,  -knee
    };

    // Wait ~1 s for controller to activate, then send once
    timer_ = this->create_wall_timer(
      1s, std::bind(&GetUpNode::send_once, this)
    );
  }

private:
  void send_once()
  {
    if (sent_) return;

    auto msg = sensor_msgs::msg::JointState();
    msg.header.stamp = now();
    msg.name     = joint_names_;
    msg.position = target_positions_;
    msg.velocity.assign(target_positions_.size(), 0.0);
    msg.effort  .assign(target_positions_.size(), 0.0);

    pub_->publish(msg);
    RCLCPP_INFO(get_logger(), "✅ get-up JointState command sent");
    sent_ = true;
    timer_->cancel();
  }

  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr                     timer_;
  bool                                             sent_;
  std::vector<std::string>                         joint_names_;
  std::vector<double>                              target_positions_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<GetUpNode>());
  rclcpp::shutdown();
  return 0;
}
