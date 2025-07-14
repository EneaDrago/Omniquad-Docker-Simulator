#include "omni_mulinex_joystic/omnimul_joy.hpp"
#include "rclcpp/executors.hpp"

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::executors::SingleThreadedExecutor exec;
    std::shared_ptr<omni_mulinex_joy::OmniMulinex_Joystic> node;
    node = std::make_shared<omni_mulinex_joy::OmniMulinex_Joystic>();
    exec.add_node(node);

    exec.spin();
    exec.remove_node(node);
    node->close();
    rclcpp::shutdown();  
    return 0;
}